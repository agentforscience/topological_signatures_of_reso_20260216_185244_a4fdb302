# Downloaded Datasets

This directory contains datasets for the research project on topological signatures of resource-constrained language model training. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: WikiText-2

### Overview
- **Source**: HuggingFace `wikitext` (wikitext-2-raw-v1)
- **Size**: 36,718 train / 3,760 validation / 4,358 test examples
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Language modeling perplexity evaluation
- **License**: Creative Commons Attribution-ShareAlike

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset.save_to_disk("datasets/wikitext-2")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext-2")
```

### Notes
- Standard benchmark for language model perplexity evaluation
- Small enough for rapid iteration during experiments
- Used to evaluate model quality at different training checkpoints

---

## Dataset 2: TinyStories

### Overview
- **Source**: HuggingFace `roneneldan/TinyStories`
- **Size**: ~2.1M stories (full dataset ~700MB)
- **Format**: HuggingFace Dataset (streaming)
- **Task**: Small-scale language model training
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
# Full download
dataset = load_dataset("roneneldan/TinyStories")
dataset.save_to_disk("datasets/tinystories")

# Or streaming for memory efficiency
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
```

### Notes
- Designed for training small language models that produce coherent stories
- Feasible to train models from scratch on limited compute
- Ideal for resource-constrained training experiments where we vary compute budgets
- Only sample data saved locally; full dataset should be streamed or downloaded separately

---

## Dataset 3: GLUE MRPC

### Overview
- **Source**: HuggingFace `nyu-mll/glue` (mrpc configuration)
- **Size**: 3,668 train / 408 validation / 1,725 test examples
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Binary classification (paraphrase detection)
- **License**: Various (see GLUE benchmark)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("nyu-mll/glue", "mrpc")
dataset.save_to_disk("datasets/glue-mrpc")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/glue-mrpc")
```

### Notes
- Standard downstream evaluation task from GLUE benchmark
- Used by Aghajanyan et al. (2020) for intrinsic dimensionality studies
- Small dataset, good for measuring generalization under resource constraints

---

## Model Checkpoints (Not Stored Here)

The following pre-trained model checkpoints with intermediate training saves are essential for this research. They are loaded on-demand via HuggingFace.

### Pythia Models (EleutherAI)
- **pythia-70m**: Smallest model, 154 checkpoints throughout training
- **pythia-160m**: Small model with full checkpoint history
- **pythia-410m**: Medium model with checkpoints

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load specific checkpoint (step 1000 of 143,000)
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m",
    revision="step1000"
)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
```

### OLMo Models (AI2)
- **OLMo-1B**: 1B parameter model with intermediate checkpoints

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
```
