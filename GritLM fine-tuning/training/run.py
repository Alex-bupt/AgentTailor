# coding: utf-8
# This file adapts MIT-licensed open-source code; the complete notice is kept in the project LICENSE.

import logging
import json
import multiprocessing
import os
from pathlib import Path
import random

import datasets
import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
)
from accelerate import Accelerator

from .arguments import CustomTrainingArguments, DataArguments, ModelArguments
from .data import CustomCollator, CustomDataset, CustomRandomSampler
from .model import ReFICRTrainModel

# ---- Prompt format tokens ----
BASE_BOS: str = "<s>"
TURN_SEP: str = "\n"

USER_BOS: str = "<|user|>\n"
USER_EOS: str = ""  # "</s>" for Zephyr-style if needed

EMBED_BOS: str = "\n<|embed|>\n"
EMBED_EOS: str = ""  # Only needed for last-token pooling

ASSISTANT_BOS: str = "\n<|assistant|>\n"
ASSISTANT_EOS: str = "</s>"

logger = logging.getLogger(__name__)
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def args_to_dtype(args):
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def filter_too_long_instructions(tokenizer, dataset, query_max_len, passage_max_len):
    def filter_fn(example):
        # Drop extreme samples to avoid slow tokenization / empty texts
        if (len(example["query"][0]) > query_max_len * 10) or not example["query"][1]:
            return False
        if (
            len(
                tokenizer.tokenize(
                    BASE_BOS + USER_BOS + example["query"][0].strip("\t\n :") + USER_EOS + EMBED_BOS
                )
            )
            >= query_max_len
        ):
            return False
        for ex in example["pos"] + example["neg"]:
            if (len(ex[0]) > passage_max_len * 10) or not ex[1]:
                return False
            if (
                len(
                    tokenizer.tokenize(
                        BASE_BOS + USER_BOS + ex[0].strip("\t\n :") + USER_EOS + EMBED_BOS
                    )
                )
                >= passage_max_len
            ):
                return False
        return True

    num_proc = max(multiprocessing.cpu_count() - 2, 1) if len(dataset) > 5000 else 1
    return dataset.filter(filter_fn, num_proc=num_proc, load_from_cache_file=True)


# ---- ZeRO-3 helpers (optional) ----
def maybe_zero_3(param):
    # Requires deepspeed.zero & ZeroParamStatus if using ZeRO-3
    if hasattr(param, "ds_id"):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return, maybe_lora_bias, lora_bias_names = {}, {}, set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                lora_bias_names.add(k.split("lora_")[0] + "bias")
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for bias_name, tensor in maybe_lora_bias.items():
            if bias_name in lora_bias_names:
                to_return[bias_name] = tensor
    else:
        raise NotImplementedError
    return {k: maybe_zero_3(v) for k, v in to_return.items()}


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    return {k: maybe_zero_3(v) for k, v in to_return.items()}


def main():
    global local_rank
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to bypass."
        )

    # logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed: %s, fp16: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    logger.info("Training args: %s", training_args)
    logger.info("Model args: %s", model_args)
    logger.info("Data  args: %s", data_args)

    set_seed(training_args.seed)

    # ---- GradCache handling ----
    gc_chunk_size = None
    if (
        (training_args.gradient_accumulation_steps > 1)
        and training_args.negatives_cross_device
        and training_args.mode in ["embedding", "unified"]
    ) or (training_args.no_gen_gas and training_args.no_emb_gas):
        gc_chunk_size = training_args.per_device_train_batch_size
        training_args.per_device_train_batch_size *= training_args.gradient_accumulation_steps
        training_args.gradient_accumulation_steps = 1
        logger.info("Using GradCache; chunk size = %d", gc_chunk_size)
    elif training_args.no_gen_gas or training_args.no_emb_gas:
        raise ValueError("no_gen_gas / no_emb_gas require GradCache")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        padding_side="right",  # mask of instruction tokens assumes right-padding
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=1,
    )
    logger.info("Config: %s", config)

    if (not tokenizer.pad_token) and tokenizer.bos_token:
        tokenizer.pad_token = tokenizer.bos_token
        logger.info("Set pad token to bos token: %s", tokenizer.pad_token)

    data_files = (
        [os.path.join(data_args.train_data, x) for x in os.listdir(data_args.train_data)]
        if os.path.isdir(data_args.train_data)
        else [data_args.train_data]
    )

    train_ds, ds_embedding_lens = [], []
    num_samples_cfg = None
    if data_args.num_samples:
        with open(data_args.num_samples, "r") as f:
            num_samples_cfg = json.load(f)

    ds_name_to_samples = {}
    if data_args.generative_max_len is None:
        data_args.generative_max_len = data_args.passage_max_len

    # ---- load & filter datasets ----
    for file in data_files:
        logger.info("Loading dataset %s", file)
        tmp_ds = datasets.load_dataset("json", data_files=file, split="train")
        tmp_len = len(tmp_ds)

        if tmp_len > data_args.max_example_num_per_dataset:
            tmp_ds = tmp_ds.select(
                random.sample(list(range(tmp_len)), data_args.max_example_num_per_dataset)
            )

        # embedding part
        if training_args.mode in ["embedding", "unified"] and "query" in tmp_ds.features:
            if isinstance(tmp_ds[0]["query"], (tuple, list)):
                logger.info("Filtering long-Instruction embedding samples in %s", file)
                tmp_ds = filter_too_long_instructions(
                    tokenizer, tmp_ds, data_args.query_max_len, data_args.passage_max_len
                )
                if num_samples_cfg:
                    key = os.path.basename(file)
                    assert key in num_samples_cfg, f"Missing num_samples for {key}"
                    tmp_len = len(tmp_ds)
                    wanted = num_samples_cfg[key]
                    if tmp_len > wanted:
                        tmp_ds = tmp_ds.select(random.sample(list(range(tmp_len)), wanted))

            ds_name_to_samples[os.path.basename(file)] = len(tmp_ds)
            train_ds.append(tmp_ds)
            continue

        # generative part
        if training_args.mode in ["unified", "generative"] and "text" in tmp_ds.features:
            if isinstance(tmp_ds[0]["text"], (tuple, list)):
                logger.info("Filtering long-Instruction generative samples in %s", file)
                num_proc = max(multiprocessing.cpu_count() - 2, 1) if tmp_len > 5000 else 1
                tmp_ds = tmp_ds.filter(
                    lambda ex: len(
                        tokenizer.tokenize(USER_BOS + ex["text"][0] + USER_EOS + ASSISTANT_BOS)
                    )
                    < data_args.generative_max_len,
                    num_proc=num_proc,
                    load_from_cache_file=True,
                )
            ds_name_to_samples[os.path.basename(file)] = len(tmp_ds)
            train_ds.append(tmp_ds)
            continue

        logger.info("Skipping dataset %s (type not recognized)", file)

    if training_args.mode == "embedding":
        ds_embedding_lens = [len(t) for t in train_ds]
        ds = datasets.concatenate_datasets(train_ds)
        logger.info("Embedding mode: %d samples", len(ds))
    elif training_args.mode == "generative":
        ds = datasets.concatenate_datasets(train_ds)
        logger.info("Generative mode: %d samples", len(ds))
    elif training_args.mode == "unified":
        ds_embedding = datasets.concatenate_datasets([t for t in train_ds if "query" in t.features])
        ds_generative = datasets.concatenate_datasets([t for t in train_ds if "text" in t.features])
        logger.info(
            "Unified mode: %d embedding samples, %d generative samples",
            len(ds_embedding),
            len(ds_generative),
        )
        for t in train_ds:
            if "query" in t.features:
                ds_embedding_lens.append(len(t))
        ds = [ds_embedding, ds_generative]
    else:
        raise NotImplementedError(training_args.mode)

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "dataset_num_samples.json"), "w") as f:
        json.dump(ds_name_to_samples, f)

    if training_args.per_device_generative_bs is not None:
        assert training_args.mode == "unified", "per_device_generative_bs only for unified mode"
        assert training_args.per_device_generative_bs < training_args.per_device_train_batch_size
        logger.info(
            "Generative batch size per device: %d", training_args.per_device_generative_bs
        )

    quantization_config, load_in_4bit = None, False
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if training_args.qlora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0:
            logging.warning("FSDP and ZeRO-3 are incompatible with QLoRA.")
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = ReFICRTrainModel(
        model_name_or_path=model_args.model_name_or_path,
        normalized=model_args.normalized,
        pooling_method=model_args.pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        in_batch_neg=training_args.in_batch_neg,
        temperature=training_args.temperature,
        mode=training_args.mode,
        projection=model_args.projection,
        attn=model_args.attn,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=args_to_dtype(training_args),
        loss_gen_type=training_args.loss_gen_type,
        loss_gen_factor=training_args.loss_gen_factor,
        use_cache=False,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        load_in_4bit=load_in_4bit,
    )

    # Special token for last-token pooling
    if model_args.pooling_method == "lasttoken":
        embed_eos_token = "</e>"
        if embed_eos_token in tokenizer.vocab:
            logger.info("Embed EOS already present: %s", embed_eos_token)
        else:
            logger.info("Adding embed EOS token: %s", embed_eos_token)
            tokenizer.add_tokens([embed_eos_token], special_tokens=True)
            model.model.resize_token_embeddings(len(tokenizer))
        config.num_vocab += 1
    else:
        embed_eos_token = EMBED_EOS

    if os.getenv("BIDIRECTIONAL_ATTN", False):
        target = model.model.model if hasattr(model.model, "model") else model.model
        target.padding_idx = tokenizer.pad_token_id

    # ---- LoRA / QLoRA ----
    if training_args.lora or training_args.qlora:
        if training_args.qlora:
            from peft import prepare_model_for_kbit_training

            model.model = prepare_model_for_kbit_training(
                model.model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            if not ddp and torch.cuda.device_count() > 1:
                model.is_parallelizable = True
                model.model_parallel = True

        from peft import get_peft_model, LoraConfig

        if training_args.gradient_checkpointing:
            model.model.enable_input_require_grads()

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model.model = get_peft_model(model.model, peft_config)
        model.model.config.use_cache = False
        model.model.print_trainable_parameters()

    train_dataset = CustomDataset(
        ds,
        args=data_args,
        tokenizer=tokenizer,
        mode=training_args.mode,
        full_bs=training_args.per_device_train_batch_size,
        generative_bs=training_args.per_device_generative_bs,
        max_seq_len=max(
            data_args.query_max_len, data_args.passage_max_len, data_args.generative_max_len
        ),
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": CustomCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
            generative_max_len=data_args.generative_max_len,
            base_bos=BASE_BOS,
            turn_sep=TURN_SEP,
            user_bos=USER_BOS,
            user_eos=USER_EOS,
            embed_bos=EMBED_BOS,
            embed_eos=embed_eos_token,
            assistant_bos=ASSISTANT_BOS,
            assistant_eos=ASSISTANT_EOS,
            prefixlm=data_args.prefixlm,
        ),
        "tokenizer": tokenizer,
    }

    if gc_chunk_size is not None:
        from .gradcache_trainer import GradCacheTrainer

        trainer = GradCacheTrainer(**trainer_kwargs)
        trainer.gc_chunk_size = gc_chunk_size
        trainer.emb_loss_fn = model.emb_loss_fn
        trainer.mode = training_args.mode
        trainer.no_gen_gas = training_args.no_gen_gas
        trainer.no_emb_gas = training_args.no_emb_gas
        trainer.split_emb = training_args.split_emb
        trainer.split_emb_full = training_args.split_emb_full
        trainer.emb_p_only = training_args.emb_p_only
        trainer.emb_q_only = training_args.emb_q_only
    else:
        trainer = Trainer(**trainer_kwargs)

    trainer.accelerator = Accelerator(mixed_precision="bf16")

    # Custom sampler when multiple embedding datasets exist
    if len(ds_embedding_lens) > 1:
        assert (
            training_args.dataloader_drop_last
        ), "Multiple datasets require --dataloader_drop_last"
        logger.info("Embedding dataset lengths: %s", ds_embedding_lens)
        total_bs = (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        )
        total_bs = total_bs * dist.get_world_size() if dist.is_initialized() else total_bs
        trainer._get_train_sampler = lambda: CustomRandomSampler(
            total_batch_size=total_bs,
            ds_lens=ds_embedding_lens,
            _num_samples=sum(ds_embedding_lens),
            data_source=train_dataset,
        )

    # Track emb/gen losses separately (unified mode)
    if training_args.mode == "unified":
        from transformers.integrations import WandbCallback
        from transformers.integrations.integration_utils import rewrite_logs
        from transformers.trainer_pt_utils import distributed_concat

        class WandbCustomCallback(WandbCallback):
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if self._wandb is None:
                    return
                if not self._initialized:
                    self.setup(args, state, model)
                if hasattr(state, "loss_emb") and hasattr(state, "loss_gen"):
                    if (
                        args.distributed_state is not None
                        and args.distributed_state.distributed_type != "NO"
                    ) or (args.distributed_state is None and args.local_rank != -1):
                        state.loss_emb = distributed_concat(state.loss_emb).mean().item()
                        state.loss_gen = distributed_concat(state.loss_gen).mean().item()
                    else:
                        state.loss_emb = state.loss_emb.mean().item()
                        state.loss_gen = state.loss_gen.mean().item()
                    if state.is_world_process_zero:
                        self._wandb.log(
                            {
                                **rewrite_logs(logs),
                                "train/global_step": state.global_step,
                                "train/loss_emb": state.loss_emb,
                                "train/loss_gen": state.loss_gen,
                            }
                        )
                    del state.loss_emb
                    del state.loss_gen
                else:
                    if state.is_world_process_zero:
                        self._wandb.log(
                            {**rewrite_logs(logs), "train/global_step": state.global_step}
                        )

        trainer.add_callback(WandbCustomCallback())

        # Patch training_step to capture both losses
        def training_step(self, model, inputs):
            model.train()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                out = self.compute_loss(model, inputs, return_outputs=True)
                loss = out[0]
                loss_emb = out[1]["loss_emb"]
                loss_gen = out[1]["loss_gen"]

            if self.args.n_gpu > 1:
                loss = loss.mean()
                loss_emb = loss_emb.mean()
                loss_gen = loss_gen.mean()

            self.accelerator.backward(loss)

            self.state.loss_emb = getattr(
                self.state, "loss_emb", torch.tensor(0.0).to(loss.device)
            )
            self.state.loss_gen = getattr(
                self.state, "loss_gen", torch.tensor(0.0).to(loss.device)
            )
            self.state.loss_emb += loss_emb.detach() / self.args.gradient_accumulation_steps
            self.state.loss_gen += loss_gen.detach() / self.args.gradient_accumulation_steps

            logger.info("loss_emb %s", loss_emb)
            logger.info("loss_gen %s", loss_gen)
            logger.info("loss %s", loss)

            return loss.detach() / self.args.gradient_accumulation_steps

        trainer.training_step = training_step.__get__(trainer)

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Train ----
    logger.info("Starting training")
    trainer.train()
    trainer.save_state()

    # Save LoRA weights (if any)
    if training_args.lora:
        state_dict = get_peft_state_maybe_zero_3(model.model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.model.named_parameters())

        if training_args.local_rank in (0, -1):
            print("Saving PEFT weights...")
            model.model.config.save_pretrained(training_args.output_dir)
            model.model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        trainer.save_model()

    # Ensure full-state save for FSDP sharded
    if trainer.is_fsdp_enabled and trainer.accelerator.state.fsdp_plugin.state_dict_type != "FULL_STATE_DICT":
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        fsd_path = os.path.join(training_args.output_dir, "full_state_dict")
        os.makedirs(fsd_path, exist_ok=True)
        trainer.save_model(fsd_path)

    # Save tokenizer & config
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        config.to_json_file(os.path.join(training_args.output_dir, "config.json"))


if __name__ == "__main__":
    main()
