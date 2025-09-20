from dataclasses import dataclass, field
import os
from typing import Optional

from transformers import TrainingArguments


# ----------------------------
# Model / Config Arguments
# ----------------------------
@dataclass
class ModelArguments:
    """Which model/config/tokenizer to fine-tune."""
    model_name_or_path: str = field(
        metadata={"help": "Path or HF hub ID of the pretrained model."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Config name or path if different from model_name_or_path."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer name or path if different from model_name_or_path."}
    )
    pooling_method: str = field(
        default="weightedmean",
        metadata={"help": "Sentence pooling: one of ['cls', 'lasttoken', 'mean', 'weightedmean']."}
    )
    normalized: bool = field(default=True, metadata={"help": "L2-normalize embeddings."})
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "Attention backend: 'eager' | 'sdpa' | 'flash_attention_2'."}
    )
    attn: str = field(
        default="bbcc",
        metadata={
            "help": (
                "Bidirectional/causal pattern for four phases (embed inst., embed sample, gen inst., gen sample). "
                "Example: 'bbcc' = bi/bi/ca/ca; 'cccc' = causal for all."
            )
        }
    )
    projection: Optional[int] = field(
        default=None,
        metadata={"help": "Optional linear projection dim for embeddings."}
    )


# ----------------------------
# Data Arguments
# ----------------------------
@dataclass
class DataArguments:
    train_data: str = field(
        default=None,
        metadata={
            "help": (
                "Path to training data (file or folder). If folder, each mini-batch is sampled from a single file, "
                "making in-batch negatives harder."
            )
        },
    )
    train_group_size: int = field(
        default=10,
        metadata={
            "help": (
                "Group size per query (1 positive + N negatives). N = train_group_size - 1. "
                "Must not exceed available negatives."
            )
        },
    )
    query_max_len: int = field(
        default=32,
        metadata={"help": "Max tokens for queries (truncate/pad)."},
    )
    passage_max_len: int = field(
        default=128,
        metadata={"help": "Max tokens for passages (truncate/pad)."},
    )
    generative_max_len: Optional[int] = field(
        default=None,
        metadata={"help": "Max tokens for generative inputs. Defaults to passage_max_len if None."},
    )
    max_example_num_per_dataset: int = field(
        default=100_000_000,
        metadata={"help": "Upper bound on examples per dataset."}
    )
    num_samples: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a JSON specifying sample counts per dataset."}
    )
    use_unique_indices: bool = field(
        default=False,
        metadata={"help": "Ensure unique samples per epoch when unified mode mixes datasets of different lengths."}
    )
    prefixlm: bool = field(
        default=False,
        metadata={"help": "Use PrefixLM for generative training."}
    )

    def __post_init__(self):
        if self.train_data is None or not os.path.exists(self.train_data):
            raise FileNotFoundError(f"Training data not found: {self.train_data}")


# ----------------------------
# Training Arguments (custom)
# ----------------------------
@dataclass
class CustomTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(
        default=False,
        metadata={"help": "Share negatives across all GPUs to increase negative pool size."}
    )
    in_batch_neg: bool = field(
        default=False,
        metadata={"help": "Enable in-batch negatives (otherwise use constructed negatives only)."}
    )
    temperature: Optional[float] = field(
        default=0.02,
        metadata={"help": "Scale similarity scores by 1/temperature before loss."}
    )
    mode: str = field(
        default="embedding",
        metadata={
            "help": (
                "One of ['unified', 'embedding', 'generative']. In 'unified', `train_data` should point to a folder "
                "containing both embedding and generative samples."
            )
        },
    )
    per_device_generative_bs: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Per-device batch size for generative part (must be <= standard batch size). "
                "Overrides grad accumulation for generative steps in unified mode."
            )
        },
    )
    no_gen_gas: bool = field(
        default=False,
        metadata={
            "help": (
                "Disable gradient accumulation for generative branch. If both emb/gen GAS are off, GradCache-style "
                "separate backward passes are used to save memory."
            )
        },
    )
    no_emb_gas: bool = field(
        default=False,
        metadata={
            "help": (
                "Disable gradient accumulation for embedding branch. See `no_gen_gas` for memory rationale."
            )
        },
    )
    loss_gen_factor: float = field(
        default=1.0,
        metadata={"help": "Scale factor for generative loss component."}
    )
    loss_gen_type: str = field(
        default="mixed",
        metadata={"help": "Generative loss type: 'mixed' or 'token'."}
    )
    lora: bool = field(default=True, metadata={"help": "Enable LoRA PEFT."})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha (scaling)."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias setting."})
    qlora: bool = field(default=False, metadata={"help": "Enable QLoRA."})
    save_safetensors: bool = field(default=False, metadata={"help": "Save checkpoints as .safetensors."})
    split_emb: bool = field(default=False, metadata={"help": "Split embedding forward/backward passes."})
    split_emb_full: bool = field(default=False, metadata={"help": "Full split of embedding forward/backward."})
    emb_q_only: bool = field(default=False, metadata={"help": "Backprop only through queries."})
    emb_p_only: bool = field(default=False, metadata={"help": "Backprop only through passages (pos/neg)."})
