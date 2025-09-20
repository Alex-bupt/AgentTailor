# GritLM Fine‑Tuning for AgentTailor

## Overview

This project contains all necessary components for fine‑tuning the GritLM model using QLoRA/LoRA and GradCache optimizations. It focuses exclusively on training utilities, model definitions, and data pipelines required for representation learning and conditional generation.

## Directory Structure

```
├── run.sh                 # Shell script to launch QLoRA/LoRA training
├── run.py                 # Training entrypoint using HuggingFace Trainer & Accelerate
├── model.py               # Definitions of Retrieval and RetrievalTrainModel with contrastive & generative losses
├── modeling_mistral.py    # Core Mistral decoder components (attention, MLP, rotary embeddings)
├── arguments.py           # Dataclass definitions for CLI arguments (Model, Data, CustomTraining)
├── data.py                # CustomDataset & CustomCollator for embedding vs. generative tasks
├── gradcache_trainer.py   # GradCacheTrainer extending HF Trainer for memory‑efficient backprop
├── utils.py               # Text processing and metric helper functions
└── requirements.txt       # Project dependencies
```

## Requirements

- **Python** 3.8+
- **Libraries** (see `requirements.txt`):
  - torch, transformers, accelerate, datasets
  - bitandbytes, peft, grad\_cache
  - pandas, tqdm

## Quickstart

1. **Single‑GPU Training**
   
   ```bash
   ./run.sh
   ```

2. **Distributed Training with Accelerate**
   
   ```bash
   python run.py \
     --model_name_or_path GritLM-7B \
     --train_data data/AgentTailor_Instruct \
     --output_dir outputs/GritLM_finetuned \
     --mode unified \
     --lora True \
     --qlora True \
     --per_device_train_batch_size 8 \
     --gradient_accumulation_steps 1 \
     --num_train_epochs 1 \
     --learning_rate 2e-5
   ```

## Component Descriptions

- **run.sh**: Wrapper for `torchrun` that sets device, QLoRA flags, and training hyperparameters.
- **run.py**: Parses CLI arguments, initializes tokenizer, config, datasets, and launches `GradCacheTrainer` with appropriate plugins.
- **model.py**: Implements:
  - Unified encoder/generator with configurable pooling and projection.
  - Extends RetrievalModel to compute contrastive and next‑token losses.
- **modeling\_mistral.py**: Provides Mistral decoder building blocks:
  - Rotary positional embeddings
  - Multi‑head attention with selective flash/SDPA support
  - Feed‑forward modules
- **arguments.py**: Defines dataclasses:
  - `ModelArguments`: Model selection and PEFT settings
  - `DataArguments`: Dataset paths, sequence lengths, batching
  - `CustomTrainingArguments`: Fine‑tuning flags (LoRa, QLoRA, GradCache, temperature, etc.)
- **data.py**: Custom data pipeline:
  - `CustomDataset`: Handles mixed embedding and generative samples
  - `CustomCollator`: Tokenization logic for contrastive and generative branches
- **gradcache\_trainer.py**: Extends HuggingFace `Trainer` to integrate GradCache for splitting backward passes and reducing memory footprint.
- **utils.py**: Utility functions for:
  - Role assignment in prompts
  - Text cleaning/pattern extraction
  - Recall\@K, NDCG\@K metric calculations
