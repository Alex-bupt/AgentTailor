# coding: utf-8
# This file adapts MIT-licensed open-source code; full notice is in the project LICENSE.

from dataclasses import dataclass
import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput  # keep for HF<4.40; use transformers.utils for newer

from ReFICR import ReFICR  # unchanged import path

logger = logging.getLogger(__name__)


# ----------------------------
# Outputs
# ----------------------------
@dataclass
class ReFICRTrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    loss_emb: Optional[Tensor] = None
    loss_gen: Optional[Tensor] = None


# ----------------------------
# Losses
# ----------------------------
class DistributedContrastiveLoss:
    def __init__(self, temperature: float, negatives_cross_device: bool, in_batch_neg: bool):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        self.in_batch_neg = in_batch_neg

        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("negatives_cross_device=True requires distributed training")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def __call__(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        if self.negatives_cross_device:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)

        scores = self.compute_similarity(q_reps, p_reps, self.in_batch_neg) / self.temperature
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target *= (p_reps.size(0) // q_reps.size(0))
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[Tensor]) -> Optional[Tensor]:
        if t is None:
            return None
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.rank] = t
        return torch.cat(all_tensors, dim=0)

    @staticmethod
    def compute_similarity(q_reps: Tensor, p_reps: Tensor, in_batch_neg: bool) -> Tensor:
        if in_batch_neg:
            return torch.matmul(q_reps, p_reps.transpose(-2, -1))
        q_reps = q_reps.unsqueeze(1)
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))


class NextTokenLoss:
    """
    loss_gen_type:
      - "token": each token equally weighted (sum / batch_size)
      - "mixed": HF default mean over tokens in batch (common mixed effect with GAS/DDP)
    """
    def __init__(self, vocab_size: int, loss_gen_type: str = "mixed", loss_gen_factor: float = 1.0):
        self.vocab_size = vocab_size
        self.loss_gen_factor = loss_gen_factor
        if loss_gen_type == "token":
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="sum")
        elif loss_gen_type == "mixed":
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError(f"Invalid loss_gen_type: {loss_gen_type}")
        self.loss_gen_type = loss_gen_type

    def __call__(self, labels: Tensor, logits: Tensor) -> Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)

        if self.loss_gen_type == "token":
            return (self.cross_entropy(shift_logits, shift_labels) / labels.size(0)) * self.loss_gen_factor
        return self.cross_entropy(shift_logits, shift_labels) * self.loss_gen_factor


# ----------------------------
# Training Model
# ----------------------------
class ReFICRTrainModel(ReFICR):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        temperature: float = 1.0,
        negatives_cross_device: bool = False,
        in_batch_neg: bool = True,
        loss_gen_type: str = "mixed",
        loss_gen_factor: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs, is_inference=False)
        self.emb_loss_fn = DistributedContrastiveLoss(temperature, negatives_cross_device, in_batch_neg)
        self.gen_add_kwargs: Dict[str, object] = {"return_dict": True}

        if "mixtral" in kwargs["model_name_or_path"].lower():
            logger.info("Using model's internal token loss (routing-aware) for Mixtral.")
            self.gen_loss_fn = None
            self.gen_add_kwargs["loss_gen_factor"] = loss_gen_factor
            self.gen_add_kwargs["output_router_logits"] = True
        else:
            self.gen_loss_fn = NextTokenLoss(
                self.model.config.vocab_size, loss_gen_type, loss_gen_factor
            )
        # Needed for accelerate/DeepSpeed integration
        self.config = self.model.config

    # ------ embedding encode ------
    def encode(self, features: Optional[Dict[str, Tensor]]):
        if features is None:
            return None

        attention_mask = features.get("attention_mask")
        instruction_lens = features.get("instruction_lens")

        kwargs = {"input_ids": features.get("input_ids"), "attention_mask": attention_mask.clone() if attention_mask is not None else None}
        if self.attn[:2] == "cb":
            kwargs["instruction_lens"] = instruction_lens
        elif self.attn[:2] == "bb":
            kwargs["is_causal"] = False

        backbone = getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model
        hidden = backbone(**kwargs)[0]

        if self.projection is not None:
            hidden = self.projection(hidden)

        # mask instruction tokens for pooling
        if instruction_lens is not None and attention_mask is not None:
            mask = attention_mask.clone()
            for i, l in enumerate(instruction_lens):
                mask[i, :l] = 0
                assert mask[i].sum() > 0, f"All zeros after masking: {mask[i]}, len={l}"
        else:
            mask = attention_mask

        reps = self.pooling(hidden, mask)
        if self.normalized:
            dtype_in = reps.dtype
            reps = torch.nn.functional.normalize(reps, dim=-1).contiguous().to(dtype_in)
        return reps.contiguous()

    # ------ forward ------
    def forward(
        self,
        query: Optional[Dict[str, Tensor]] = None,
        passage: Optional[Dict[str, Tensor]] = None,
        generative: Optional[Dict[str, Tensor]] = None,
        q_reps: Optional[Tensor] = None,
        p_reps: Optional[Tensor] = None,
        q_grad: bool = True,
        p_grad: bool = True,
    ) -> ReFICRTrainOutput:

        # generative first
        if generative is not None:
            if self.gen_loss_fn is not None:
                labels = generative.pop("labels")
                logits = self.model(**generative, **self.gen_add_kwargs).logits
                loss_gen = self.gen_loss_fn(labels, logits)
            else:
                loss_gen = self.model(**generative, **self.gen_add_kwargs).loss
        else:
            loss_gen = None

        # encode query / passage if not provided
        if q_reps is None and query is not None:
            q_reps = self.encode(query) if q_grad else torch.no_grad()(lambda: self.encode(query))()
        if p_reps is None and passage is not None:
            p_reps = self.encode(passage) if p_grad else torch.no_grad()(lambda: self.encode(passage))()

        loss_emb = (
            self.emb_loss_fn(q_reps, p_reps)
            if (q_reps is not None and p_reps is not None)
            else None
        )

        loss = sum(x for x in (loss_emb, loss_gen) if x is not None)

        return ReFICRTrainOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=loss,
            loss_emb=loss_emb,
            loss_gen=loss_gen,
        )

    # passthrough
    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)
