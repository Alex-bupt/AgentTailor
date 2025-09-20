# coding: utf-8
# This file adapts open-source code released under the MIT License.
# The original notice is kept in the project LICENSE; only minimal excerpts are shown here for anonymity.

from dataclasses import dataclass
import logging
import math
import random
from typing import Iterator, List, Tuple, Union

import datasets
import torch
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments  # keep your existing import path

logger = logging.getLogger(__name__)


# ----------------------------
# Dataset
# ----------------------------
class CustomDataset(torch.utils.data.Dataset):
    """
    Handles three modes:
      - 'embedding': only embedding data
      - 'generative': only generative data
      - 'unified': both (expects a list/tuple: [embedding_ds, generative_ds])

    Optionally maintains per-process unique index pools when dataset sizes differ.
    """

    def __init__(
        self,
        dataset: Union[datasets.Dataset, List[datasets.Dataset]],
        args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        mode: str = "embedding",
        full_bs: int = None,
        generative_bs: int = None,
        max_seq_len: int = 2048,
    ):
        self.indices_emb, self.indices_gen = None, None

        if mode == "unified":
            self.ds_embedding = dataset[0]
            self.ds_generative = dataset[1]
            self.len_embedding = len(self.ds_embedding)
            self.len_generative = len(self.ds_generative)
            self.total_len = max(self.len_embedding, self.len_generative)
            if args.use_unique_indices:
                self._set_indices()
        elif mode == "embedding":
            self.ds_embedding = dataset
            self.total_len = self.len_embedding = len(self.ds_embedding)
        elif mode == "generative":
            self.ds_generative = dataset
            self.total_len = self.len_generative = len(self.ds_generative)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        # Rough safeguard: truncate overly long strings before tokenization
        self.max_char_len = max_seq_len * 10

        self.n_samples = self.total_len * full_bs
        if generative_bs is not None:
            assert full_bs >= generative_bs, "full_bs must be >= generative_bs"
            assert full_bs % generative_bs == 0, "full_bs must be divisible by generative_bs"
            self.take_nth = full_bs // generative_bs
        else:
            self.take_nth = 1

    def _set_indices(self):
        """
        When embedding / generative datasets have different sizes in unified mode,
        keep a per-process set of indices to ensure random sampling across epochs.
        """
        if self.len_embedding > self.len_generative:
            idx = list(range(self.len_generative))
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                idx = idx[rank::world_size]
            self.indices_gen = set(idx)
        elif self.len_embedding < self.len_generative:
            idx = list(range(self.len_embedding))
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                idx = idx[rank::world_size]
            self.indices_emb = set(idx)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding], BatchEncoding]:
        """
        Returns (query, passages, generative_text). All may contain strings or lists of strings.
        Note: if training >1 epoch in unified mode, repeat data externally to avoid fixed pairing.
        """
        query, passages, generative = None, None, None

        # -------- Embedding branch --------
        if self.mode in ["unified", "embedding"]:
            if self.indices_emb is not None:
                if not self.indices_emb:
                    self._set_indices()
                item = self.indices_emb.pop()
            elif item >= self.len_embedding:
                item = random.randint(0, self.len_embedding - 1)

            query = self.ds_embedding[item]["query"]
            query = self._truncate(query)

            passages = []
            pos = random.choice(self.ds_embedding[item]["pos"])
            pos = self._truncate(pos)
            passages.append(pos)

            neg_pool = self.ds_embedding[item]["neg"]
            needed = self.args.train_group_size - 1
            if len(neg_pool) < needed:
                mult = math.ceil(needed / len(neg_pool))
                negs = random.sample(neg_pool * mult, needed)
            else:
                negs = random.sample(neg_pool, needed)

            passages.extend(self._truncate_batch(negs))

        # -------- Generative branch --------
        if (self.mode in ["unified", "generative"]) and (self.n_samples % self.take_nth == 0):
            if self.indices_gen is not None:
                if not self.indices_gen:
                    self._set_indices()
                item = self.indices_gen.pop()
            elif item >= self.len_generative:
                item = random.randint(0, self.len_generative - 1)
            generative = self.ds_generative[item]["text"]

        self.n_samples -= 1
        return query, passages, generative

    # ---- helpers ----
    def _truncate(self, x):
        if isinstance(x, str):
            return x[:self.max_char_len]
        if isinstance(x, list):
            return [t[:self.max_char_len] for t in x]
        raise ValueError(f"Unexpected type: {type(x)}")

    def _truncate_batch(self, items):
        out = []
        for it in items:
            if isinstance(it, str):
                out.append(it[:self.max_char_len])
            elif isinstance(it, list):
                out.append([t[:self.max_char_len] for t in it])
            else:
                raise ValueError(f"Unexpected type: {type(it)}")
        return out


# ----------------------------
# Collator
# ----------------------------
@dataclass
class CustomCollator(DataCollatorWithPadding):
    """
    Wraps tokenizer calls for (query, passages, generative) triples. Handles:
    - optional instruction/text tuples
    - multi-turn generative formatting
    - masking labels for prefix-LM or instruction segments
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    generative_max_len: int = 128

    # Prompt pieces
    base_bos: str = ""
    turn_sep: str = ""
    user_bos: str = ""
    user_eos: str = ""
    embed_bos: str = ""
    embed_eos: str = ""  # used only if you rely on last-token pooling
    assistant_bos: str = ""
    assistant_eos: str = ""

    prefixlm: bool = False  # if True, mask everything up to last assistant turn

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        generative = [f[2] for f in features]

        # Flatten nested passage lists
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        batch = {}
        q_instr_lens = None
        d_instr_lens = None
        g_instr_lens = None

        # -------- embedding side: (instruction, text) tuples --------
        if query and isinstance(query[0], (tuple, list)):
            q_instr_lens = [
                len(
                    self.tokenizer.tokenize(
                        self.base_bos + self.user_bos + f[0].strip("\t\n :") + self.user_eos + self.embed_bos
                        if f[0].strip("\t\n :")
                        else self.base_bos + self.embed_bos.lstrip()
                    )
                )
                for f in query
            ]
            d_instr_lens = [
                len(
                    self.tokenizer.tokenize(
                        self.base_bos + self.user_bos + f[0].strip("\t\n :") + self.user_eos + self.embed_bos
                        if f[0].strip("\t\n :")
                        else self.base_bos + self.embed_bos.lstrip()
                    )
                )
                for f in passage
            ]

            query = [
                (
                    self.base_bos + self.user_bos + f[0].strip("\t\n :") + self.user_eos + self.embed_bos + f[1] + self.embed_eos
                    if f[0].strip("\t\n :")
                    else self.base_bos + self.embed_bos.lstrip() + f[1] + self.embed_eos
                )
                for f in query
            ]
            passage = [
                (
                    self.base_bos + self.user_bos + f[0].strip("\t\n :") + self.user_eos + self.embed_bos + f[1] + self.embed_eos
                    if f[0].strip("\t\n :")
                    else self.base_bos + self.embed_bos.lstrip() + f[1] + self.embed_eos
                )
                for f in passage
            ]

        # -------- generative side: (u1, a1, u2, a2, ... ) --------
        if generative and isinstance(generative[0], (tuple, list)):
            g_instr_lens = [
                [
                    len(
                        self.tokenizer.tokenize(
                            (self.base_bos if i == 0 else "") + self.user_bos + z + self.user_eos + self.assistant_bos
                        )
                    )
                    if i % 2 == 0
                    else len(self.tokenizer.tokenize(z.strip() + self.assistant_eos))
                    for i, z in enumerate(f[:-1])
                ]
                for f in generative
                if f is not None
            ]
            generative = [
                self.base_bos
                + self.turn_sep.join(
                    [
                        self.user_bos + f[i] + self.user_eos + self.assistant_bos + f[i + 1].strip() + self.assistant_eos
                        for i in range(0, len(f), 2)
                    ]
                )
                for f in generative
                if f is not None
            ]

        # -------- tokenize --------
        if query[0] is not None:
            batch["query"] = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=self.query_max_len,
                return_tensors="pt",
                add_special_tokens=False,  # special tokens already in prompt
            )
            batch["passage"] = self.tokenizer(
                passage,
                padding=True,
                truncation=True,
                max_length=self.passage_max_len,
                return_tensors="pt",
                add_special_tokens=False,
            )

        if generative[0] is not None:
            batch["generative"] = self.tokenizer(
                generative,
                padding=True,
                truncation=True,
                max_length=self.generative_max_len,
                return_tensors="pt",
                add_special_tokens=False,
            )
            batch["generative"]["labels"] = batch["generative"]["input_ids"].clone()
            # mask pads in labels (leave token 0 untouched in case it's pad/bos)
            labels = batch["generative"]["labels"]
            labels[:, 1:][labels[:, 1:] == self.tokenizer.pad_token_id] = -100

        # -------- mask instruction tokens for embedding --------
        if q_instr_lens:
            for i, l in enumerate(q_instr_lens):
                assert batch["query"]["input_ids"][i, l] != self.tokenizer.pad_token_id, f"No text to embed: {query[i]}"
            for i, l in enumerate(d_instr_lens):
                assert batch["passage"]["input_ids"][i, l] != self.tokenizer.pad_token_id, f"No text to embed: {passage[i]}"
            batch["query"]["instruction_lens"] = torch.tensor(q_instr_lens)
            batch["passage"]["instruction_lens"] = torch.tensor(d_instr_lens)

        # -------- mask instruction tokens for generative --------
        if g_instr_lens:
            for i, lens in enumerate(g_instr_lens):
                cur = 0
                for j, l in enumerate(lens):
                    # Mask user turns (even idx) and, if prefixLM, also assistant turns (odd idx before final)
                    if (j % 2 == 0) or self.prefixlm:
                        batch["generative"]["labels"][i, cur : cur + l] = -100
                    cur += l

        return batch


# ----------------------------
# Sampler
# ----------------------------
@dataclass
class CustomRandomSampler(torch.utils.data.sampler.RandomSampler):
    """
    Ensures, as much as possible, that each batch is formed from a single dataset
    when concatenating multiple datasets with different lengths.
    """
    total_batch_size: int = 8
    ds_lens: List[int] = None
    _num_samples: int = None
    data_source: CustomDataset = None
    replacement: bool = False

    def __iter__(self) -> Iterator[int]:
        if not hasattr(self, "generator") or self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = self.generator

        # 1) Shuffle indices within each dataset
        ds_indices = [torch.randperm(n, generator=generator).tolist() for n in self.ds_lens]
        # 2) Offset each sub-list to match the concatenated dataset indexing
        ds_indices = [[idx + sum(self.ds_lens[:j]) for idx in ds_indices[j]] for j in range(len(self.ds_lens))]
        # 3) Split into (almost) pure batches
        ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]

        # Collect incomplete batches to later form mixed ones
        incomplete = []
        for b in ds_batches:
            if len(b[-1]) < self.total_batch_size:
                incomplete.append(b.pop())

        if incomplete:
            order = torch.randperm(len(incomplete), generator=generator).tolist()
            incomplete = torch.cat([incomplete[i] for i in order])
            mixed = list(torch.split(incomplete, self.total_batch_size))
            if len(mixed[-1]) < self.total_batch_size:
                mixed.pop()
            ds_batches = sum(ds_batches, []) + mixed
            logger.info(
                f"Global BS={self.total_batch_size}: {len(ds_batches) - len(mixed)} single-dataset batches, "
                f"{len(mixed)} mixed batches."
            )
        else:
            ds_batches = sum(ds_batches, [])
            logger.info(f"Global BS={self.total_batch_size}: {len(ds_batches)} single-dataset batches.")

        # Shuffle batch order and yield flattened indices
        order = torch.randperm(len(ds_batches), generator=generator).tolist()
        flat = torch.cat([ds_batches[i] for i in order]).tolist()
        yield from (int(i) for i in flat)
