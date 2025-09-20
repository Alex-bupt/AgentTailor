# coding: utf-8
# This file adapts MIT-licensed open-source code. The original notice is kept in the project LICENSE.

import math
import os
import time
from typing import Optional

import numpy as np
from packaging import version
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader

from transformers import Trainer
from transformers.debug_utils import DebugOption
from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
    is_deepspeed_available,
)
from transformers.utils import (
    logging,
    is_accelerate_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_datasets_available,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    TrainOutput,
    has_length,
    speed_metrics,
    seed_worker,
)

from grad_cache import GradCache  # unchanged import

if is_datasets_available():
    import datasets

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
    )

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

logger = logging.get_logger(__name__)


class GradCacheTrainer(Trainer):
    # ----------------------------
    # Accelerator setup
    # ----------------------------
    def create_accelerator_and_postprocess(self):
        from datetime import timedelta
        from accelerate.utils import InitProcessGroupKwargs

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))  # 10 hours
        grad_acc_kwargs = {
            "num_steps": self.args.gradient_accumulation_steps,
            "sync_with_dataloader": False,
        }
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        self.accelerator = Accelerator(
            dispatch_batches=self.args.dispatch_batches,
            split_batches=self.args.split_batches,
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            kwargs_handlers=[kwargs],
        )
        self.gather_function = self.accelerator.gather_for_metrics
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "FSDP activation_checkpointing and gradient_checkpointing cannot both be True."
                    )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    # ----------------------------
    # Checkpointing
    # ----------------------------
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and os.listdir(output_dir):
            logger.warning(
                f"Checkpoint dir {output_dir} already exists and is non-empty. Proceeding anyway."
            )
            staging_output_dir = output_dir
        else:
            staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")

        self.save_model(staging_output_dir, _internal_call=True)

        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(staging_output_dir)
            self._save_rng_state(staging_output_dir)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]
            op = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or op(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        if self.args.should_save:
            self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(staging_output_dir)

        self.args.distributed_state.wait_for_everyone()
        if staging_output_dir != output_dir:
            with self.args.main_process_first(
                desc="Renaming model checkpoint folder", local=self.args.save_on_each_node
            ):
                if os.path.exists(staging_output_dir):
                    if torch.distributed.is_initialized():
                        if torch.distributed.get_rank() == 0:
                            os.rename(staging_output_dir, output_dir)
                        torch.distributed.barrier()
                    else:
                        os.rename(staging_output_dir, output_dir)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    # ----------------------------
    # Helper for no-GAS losses
    # ----------------------------
    def get_loss_no_gas(self, *args, **kwargs):
        model = kwargs.pop("model")
        get_preps = kwargs.pop("get_preps", False)
        loss_mult = kwargs.pop("loss_mult", None)
        with self.compute_loss_context_manager():
            if get_preps:
                out = model(*args, **kwargs)
                loss, reps = out.loss, out.p_reps
            else:
                loss = model(*args, **kwargs).loss
            if loss_mult is not None:
                loss *= loss_mult

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        if get_preps:
            return loss.detach(), reps.detach()
        return loss.detach()

    # ----------------------------
    # Training loop (modified HuggingFace logic)
    # ----------------------------
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Training with batch size: {self._train_batch_size}")
        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = (
            self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = max(len_dataloader // args.gradient_accumulation_steps, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps)
                        * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = 2 ** 31 - 1  # large proxy
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader, args.max_steps)
                    * args.gradient_accumulation_steps
                )
        else:
            raise ValueError("args.max_steps must be > 0 if dataloader has no length")

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                raise ValueError("--debug underflow_overflow not supported under DP. Use DDP.")
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
        )

        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # absolute values for logging/eval/save steps
        if args.logging_steps is not None:
            self.state.logging_steps = (
                math.ceil(max_steps * args.logging_steps)
                if args.logging_steps < 1
                else args.logging_steps
            )
        if args.eval_steps is not None:
            self.state.eval_steps = (
                math.ceil(max_steps * args.eval_steps)
                if args.eval_steps < 1
                else args.eval_steps
            )
        if args.save_steps is not None:
            self.state.save_steps = (
                math.ceil(max_steps * args.save_steps)
                if args.save_steps < 1
                else args.save_steps
            )

        if args.gradient_checkpointing:
            gckw = args.gradient_checkpointing_kwargs or {}
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gckw)

        model = self._wrap_model(self.model_wrapped)
        use_accelerator_prepare = model is self.model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        if model is not self.model:
            self.model_wrapped = model

        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Adjusted batch size (DataParallel) = {self._train_batch_size:,}")
        logger.info(f"  Total train batch size = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        # ---- GradCache setup ----
        if not self.no_emb_gas:
            dtype = None
            if self.args.bf16:
                dtype = torch.bfloat16
            elif self.args.fp16:
                dtype = torch.float16

            if os.getenv("BF16", False):
                gc = GradCache(
                    models=[model, model, model],
                    chunk_sizes=self.gc_chunk_size,
                    loss_fn=self.emb_loss_fn,
                    get_rep_fn=lambda x: x["q_reps"].to(dtype=dtype) if dtype else x["q_reps"],
                )
            else:
                gc = GradCache(
                    models=[model, model, model],
                    chunk_sizes=self.gc_chunk_size,
                    loss_fn=self.emb_loss_fn,
                    get_rep_fn=lambda x: x["q_reps"],
                )

            def model_call(self_gc, m, m_in):  # keep signature
                return m(m_in)

            gc.model_call = model_call.__get__(gc)
            no_sync_except_last = torch.distributed.is_initialized()

        if self.no_emb_gas or self.no_gen_gas:
            assert self.accelerator.gradient_accumulation_steps == 1, "GAS must be 1 here."

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        if resume_from_checkpoint and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = (
                    self.state.global_step % num_update_steps_per_epoch
                ) * args.gradient_accumulation_steps
            logger.info("  Resuming training from checkpoint")
            logger.info(f"  Resume epoch {epochs_trained}, global step {self.state.global_step}")

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if not args.ignore_data_skip:
            for _ in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    kinds.append(SeedableRandomSampler)
                is_random = isinstance(sampler, tuple(kinds))
                if is_torch_less_than_1_11 or not is_random:
                    for __ in train_dataloader:
                        break
                else:
                    sampler = sampler or []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    name = getattr(self.model, "main_input_name", "input_ids")
                    if name not in inputs:
                        logger.warning(
                            "Can't track num_input_tokens_seen: missing `main_input_name`."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(
                            inputs[name]
                        ).numel()

                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                with self.accelerator.accumulate(model):
                    # ---------------- MODIFIED BLOCK ----------------
                    model.train()
                    inputs = self._prepare_inputs(inputs)

                    # generative first
                    if self.mode in ["unified", "generative"]:
                        if self.no_gen_gas:
                            loss_gen = self.get_loss_no_gas(
                                model=model, generative=inputs["generative"]
                            )
                        else:
                            loss_gen = torch.zeros((), device=args.device)
                            chunks = gc.split_inputs(inputs["generative"], self.gc_chunk_size)
                            for chunk in chunks:
                                loss_chunk = model(generative=chunk).loss_gen / len(chunks)
                                loss_chunk.backward()
                                loss_gen += loss_chunk.detach()

                    # embedding
                    if self.mode in ["unified", "embedding"]:
                        if self.split_emb:
                            loss_emb_p, p_reps = self.get_loss_no_gas(
                                model=model,
                                query=inputs["query"],
                                passage=inputs["passage"],
                                q_grad=False,
                                get_preps=True,
                            )
                            loss_emb_q = self.get_loss_no_gas(
                                model=model,
                                query=inputs["query"],
                                p_reps=p_reps,
                                p_grad=False,
                            )
                            assert torch.allclose(loss_emb_q, loss_emb_p)
                            loss_emb = loss_emb_q

                        elif self.split_emb_full:
                            with self.compute_loss_context_manager():
                                out = model(
                                    query=inputs["query"],
                                    passage=inputs["passage"],
                                    q_grad=False,
                                    pos_grad=False,
                                )
                                loss, q_reps, p_reps = out.loss, out.q_reps, out.p_reps
                                p_reps = p_reps.detach()

                            if self.args.n_gpu > 1:
                                loss = loss.mean()
                            self.accelerator.backward(loss)
                            loss = loss.detach()
                            q_reps = q_reps.detach()

                            with self.compute_loss_context_manager():
                                loss2 = model(
                                    q_reps=q_reps,
                                    passage=inputs["passage"],
                                    p_reps=p_reps,
                                    q_grad=False,
                                    neg_grad=False,
                                ).loss
                            if self.args.n_gpu > 1:
                                loss2 = loss2.mean()
                            self.accelerator.backward(loss2)
                            loss2 = loss2.detach()

                            with self.compute_loss_context_manager():
                                loss3 = model(
                                    query=inputs["query"],
                                    p_reps=p_reps,
                                    pos_grad=False,
                                    neg_grad=False,
                                ).loss
                            if self.args.n_gpu > 1:
                                loss3 = loss3.mean()
                            self.accelerator.backward(loss3)
                            loss_emb = loss3.detach()

                        elif self.emb_q_only:
                            loss_emb = self.get_loss_no_gas(
                                model=model,
                                query=inputs["query"],
                                passage=inputs["passage"],
                                p_grad=False,
                            )
                        elif self.emb_p_only:
                            loss_emb = self.get_loss_no_gas(
                                model=model,
                                query=inputs["query"],
                                passage=inputs["passage"],
                                q_grad=False,
                            )
                        elif self.no_emb_gas:
                            loss_emb = self.get_loss_no_gas(
                                model=model, query=inputs["query"], passage=inputs["passage"]
                            )
                        else:
                            loss_emb = gc(
                                inputs["query"],
                                inputs["passage"],
                                no_sync_except_last=no_sync_except_last,
                            )

                    # sum losses
                    if self.mode == "unified":
                        tr_loss_step = loss_emb + loss_gen
                        self.state.loss_emb = getattr(
                            self.state, "loss_emb", torch.tensor(0.0).to(loss_emb.device)
                        )
                        self.state.loss_gen = getattr(
                            self.state, "loss_gen", torch.tensor(0.0).to(loss_emb.device)
                        )
                        self.state.loss_emb += loss_emb
                        self.state.loss_gen += loss_gen
                    elif self.mode == "embedding":
                        tr_loss_step = loss_emb
                    else:
                        tr_loss_step = loss_gen
                    # ---------------- MODIFIED BLOCK END ----------------

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_small_epoch = steps_in_epoch <= args.gradient_accumulation_steps and (
                    step + 1
                ) == steps_in_epoch

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or is_last_small_epoch
                ):
                    if is_last_small_epoch:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    if args.max_grad_norm and args.max_grad_norm > 0:
                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
                        else:
                            self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run and not isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "Empty epoch_iterator; stopping at step "
                    f"{self.state.global_step}. Likely with IterableDataset & high max_steps."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "TPU debug metrics enabled but no TPU found. Check config."
                    )

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining complete.\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()
            self._load_best_model()

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to save_total_limit=1")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        self._finish_current_push()

        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
