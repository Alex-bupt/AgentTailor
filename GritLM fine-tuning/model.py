# coding: utf-8
# This file contains adaptations of an open-source implementation released under the MIT License.
# The original copyright and permission notice are included in the project LICENSE file.

from typing import Dict, List, Union, cast
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralForCausalLM,
    MistralConfig,
    MistralPreTrainedModel,
)

# ----------------------------
# Config & Model Definitions
# ----------------------------
class RetrievalModelConfig(MistralConfig):
    model_type = "RetrievalModel"


class RetrievalModel(MistralPreTrainedModel):
    """
    Unified model for both embedding and generation modes, with configurable pooling and (optional) projection.
    """
    config_class = RetrievalModelConfig

    def __init__(
        self,
        model_name_or_path: str = None,
        mode: str = "unified",                  # ['unified', 'embedding', 'generative']
        pooling_method: str = "mean",           # ['cls', 'lasttoken', 'mean', 'weightedmean']
        normalized: bool = True,
        projection: int = None,
        is_inference: bool = True,
        embed_eos: str = "",
        attn: str = "bbcc",                     # ['bbcc', 'cccc', 'bb', 'cc']
        config: MistralConfig = RetrievalModelConfig(),
        **kwargs,
    ):
        super().__init__(config)

        # ---- Backbone selection ----
        if mode == "embedding":
            # Some encoder-only checkpoints need explicit class
            if any(t in model_name_or_path for t in ["gtr", "t5", "instructor"]):
                from transformers import T5EncoderModel
                self.model = T5EncoderModel.from_pretrained(model_name_or_path, **kwargs)
            else:
                self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
            self.embedding_attr = None
        else:
            # CausalLM backbones
            self.model = MistralForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            self.generate = self.model.generate
            if hasattr(self.model, "model"):
                self.embedding_attr = "model"          # LLaMA/Mistral style
            elif hasattr(self.model, "transformer"):
                self.embedding_attr = "transformer"    # GPT-Neo/J style
            else:
                raise ValueError("Cannot locate embedding attribute inside the loaded model.")

        # ---- Optional projection head ----
        self.projection = (
            torch.nn.Linear(self.model.config.hidden_size, int(projection), dtype=self.model.dtype)
            if projection is not None
            else None
        )

        self.normalized = normalized
        self.pooling_method = pooling_method
        self.embed_eos = embed_eos
        self.attn = attn
        if self.attn not in [None, "bbcc", "cccc", "bb", "cc"]:
            raise ValueError(f"Unsupported attention setting: {self.attn}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = 1

        if is_inference:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
            if (self.tokenizer.pad_token is None) and (self.tokenizer.eos_token is not None):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.embed_eos:
                assert self.embed_eos in self.tokenizer.vocab, f"EOS token '{self.embed_eos}' not found."

            self.model.eval()
            if "device_map" not in kwargs:
                self.model.to(self.device)
                if mode == "embedding":
                    self.num_gpus = torch.cuda.device_count()
                    if self.num_gpus > 1:
                        self.model = torch.nn.DataParallel(self.model)

        print(
            f"RetrievalModel initialized: dtype={self.model.dtype}, pool={self.pooling_method}, "
            f"mode={mode}, attn={self.attn}"
        )

    # ----------------------------
    # Public API
    # ----------------------------
    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Encode queries for retrieval/reranking tasks."""
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self,
        corpus: Union[List[str], str, List[Dict[str, str]]],
        **kwargs,
    ) -> np.ndarray:
        """Encode a corpus for retrieval tasks."""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and corpus and isinstance(corpus[0], dict):
            corpus = [
                (doc.get("title", "") + " " + doc["text"]).strip() if "text" in doc else ""
                for doc in corpus
            ]
        return self.encode(corpus, **kwargs)

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        instruction: str = "",
        embed_instruction: bool = False,
        get_cache: bool = False,
        convert_to_tensor: bool = False,
        recast: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor, tuple]:
        """
        Vectorize sentences with pooling + optional normalization & projection.
        Returns:
            np.ndarray (default) or torch.Tensor if convert_to_tensor=True.
            If get_cache=True, returns (embeddings, kv_cache).
        """
        if self.num_gpus > 1:
            batch_size *= self.num_gpus

        single_input = isinstance(sentences, str)
        if single_input:
            sentences = [sentences]

        embeddings_list, kv_cache_list = [], []

        for start in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences) < 256):
            batch_sents = [instruction + s + self.embed_eos for s in sentences[start:start + batch_size]]

            inputs = self.tokenizer(
                batch_sents,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            ).to(self.device)

            if (self.attn is not None) and self.attn.startswith("bb"):
                inputs["is_causal"] = False
            if get_cache:
                inputs["use_cache"] = True

            backbone = getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model
            outputs = backbone(**inputs)

            hidden = outputs[0]
            if get_cache:
                assert not kv_cache_list, "Cache collection supports only one batch."
                kv_cache_list = outputs[1]

            if self.projection is not None:
                hidden = self.projection(hidden)

            # Mask out instruction tokens for mean pooling if needed
            if instruction and (not embed_instruction) and ("mean" in self.pooling_method):
                instr_tokens = self.tokenizer(
                    instruction,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                )["input_ids"]
                inputs["attention_mask"][:, : len(instr_tokens)] = 0

            pooled = self.pooling(hidden, inputs["attention_mask"], recast=recast)

            if self.normalized:
                dtype_in = pooled.dtype
                pooled = torch.nn.functional.normalize(pooled, dim=-1).to(dtype_in)

            pooled = cast(torch.Tensor, pooled)
            if convert_to_tensor:
                embeddings_list.append(pooled)
            else:
                embeddings_list.append(pooled.cpu().to(torch.float32).numpy())

        embeddings = (
            torch.cat(embeddings_list, dim=0) if convert_to_tensor else np.concatenate(embeddings_list, axis=0)
        )
        if single_input:
            embeddings = embeddings[0]

        if get_cache:
            return embeddings, kv_cache_list
        return embeddings

    # ----------------------------
    # Pooling
    # ----------------------------
    def pooling(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        recast: bool = False,
    ) -> torch.Tensor:
        """
        hidden_state: [B, N, D]
        attention_mask: [B, N]
        """
        hidden_state = hidden_state.to(attention_mask.device)

        if self.pooling_method == "cls":
            embedding = hidden_state[:, 0]

        elif self.pooling_method == "lasttoken":
            b, n, d = hidden_state.size()
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_rev = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_idx = attention_mask.size(1) - argmax_rev - 1
            gather_idx = torch.clamp(gather_idx, min=0)

            gather_idx = gather_idx.unsqueeze(-1).repeat(1, d).unsqueeze(1)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_idx).squeeze(1)

        elif self.pooling_method in ["mean", "weightedmean"]:
            mask = attention_mask.float()
            if self.pooling_method == "weightedmean":
                mask *= mask.cumsum(dim=1)
            sum_vec = torch.sum(hidden_state * mask.unsqueeze(-1), dim=1)
            denom = mask.sum(dim=1, keepdim=True)
            embedding = sum_vec / denom

        else:
            raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")

        return embedding.to(hidden_state.dtype) if recast else embedding
