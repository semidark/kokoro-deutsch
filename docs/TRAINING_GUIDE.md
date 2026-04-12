# Fine-Tuning Kokoro-82M for a New Language

This guide documents the full process of fine-tuning [Kokoro-82M](https://github.com/hexgrad/kokoro) on a new language using StyleTTS2's training code. It is written from direct experience producing a German single-speaker model on a local AMD GPU (no cloud required), and covers every non-obvious decision, gotcha, and bug fix encountered along the way.

The model weights and training data cannot be shared, but the approach is fully replicable.

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware & Prerequisites](#hardware--prerequisites)
3. [Step 1: Dataset Preparation](#step-1-dataset-preparation)
4. [Step 2: Weight Conversion](#step-2-weight-conversion)
5. [Step 3: Symbol Mapping (Critical)](#step-3-symbol-mapping-critical)
6. [Step 4: StyleTTS2 Environment Setup](#step-4-styletts2-environment-setup)
7. [Step 4.5: weight_norm API Migration (Critical)](#step-45-weight_norm-api-migration-critical)
8. [Step 5: Config File](#step-5-config-file)
9. [Step 6: Smoke Testing](#step-6-smoke-testing)
10. [Step 7: Training (Stage 1)](#step-7-training)
11. [Step 7.5: Stage 2 Training (Prosody Predictor)](#step-75-stage-2-training-prosody-predictor)
12. [Step 8: Voicepack Extraction](#step-8-voicepack-extraction)
13. [AMD ROCm Notes](#amd-rocm-notes)
14. [Known Issues & Fixes in StyleTTS2](#known-issues--fixes-in-styletts2)
15. [Phoneme Compatibility Reference](#phoneme-compatibility-reference)

---

## Overview

Kokoro-82M is a StyleTTS 2-based TTS model. Its training code lives in the [StyleTTS2](https://github.com/yl4579/StyleTTS2) repository, not in the Kokoro repo itself. Fine-tuning involves:

1. Preparing a dataset in the format StyleTTS2 expects (WAV + IPA phoneme lists)
2. Converting Kokoro's published HuggingFace weights to StyleTTS2's checkpoint format
3. Patching StyleTTS2's symbol table to match Kokoro's pre-trained token indices
4. **Migrating StyleTTS2's weight_norm/spectral_norm API** to match Kokoro's inference code
5. Running Stage 1 fine-tuning (`train_first.py`) for ~10 epochs (decoder + alignment)
6. Running Stage 2 fine-tuning (`train_second.py`) for ~10 epochs (prosody predictor)
7. Extracting a voicepack and loading into Kokoro's `KPipeline`

**What you end up with:** a fine-tuned `.pth` checkpoint directly loadable by `KModel`, plus a `.pt` voicepack for use with `KPipeline`.

**What this guide does not cover:** training from scratch, diffusion model training, or multi-speaker setups.

## Status: Both Stages Now Fully Functional

**April 2026:** This guide documents a **fully working, end-to-end training pipeline** for fine-tuning Kokoro-82M on any language. Previous issues with Stage 2 training have been resolved. The five critical fixes are:

1. **DataParallel wrapping order**: Moved `MyDataParallel` wrapping to *after* checkpoint loading so state dict keys match correctly
2. **Missing pretrained weights**: Removed `bert`, `bert_encoder`, `predictor` from `ignore_modules` so Stage 2 actually loads pretrained English Kokoro weights
3. **Missing adversarial logic**: Restored accidentally deleted `y_rec_gt` / `y_rec_gt_pred` computation for `joint_epoch` training
4. **GAN discriminator activation**: Fixed discriminator activation to gate on `joint_epoch` instead of `diff_epoch`
5. **Diffusion sampler bypass**: Added `diffusion_enabled` flag to use ground-truth style vectors when diffusion is disabled

All fixes are committed to the `semidark/StyleTTS2` submodule. Clone with `--recurse-submodules` to get the patched version.

---

## Hardware & Prerequisites

### GPU Requirements

| Hardware | Status | Notes |
|----------|--------|-------|
| NVIDIA (CUDA) | Recommended | Any GPU with 10GB+ VRAM. batch_size=4 works on 12GB. |
| AMD (ROCm) | Works with caveats | Tested on Radeon 8060S (Strix Halo) with ROCm 7.12. See [AMD ROCm Notes](#amd-rocm-notes). |
| CPU only | Not practical | Training would take weeks. |

### System Dependencies

```bash
# Required for audio processing and G2P
# Ubuntu/Debian:
sudo apt-get install espeak-ng libsndfile1

# macOS:
brew install espeak-ng libsndfile
```

Both are mandatory. `espeak-ng` drives the IPA phonemization (`misaki`). `libsndfile` is required by `soundfile` for WAV I/O.

### Python Environment

This project uses `uv`. **Python 3.10–3.13 required** (tested on 3.13.12; see `requires-python` in `pyproject.toml`).

The training environment for StyleTTS2 is separate and uses a venv:

```bash
# For StyleTTS2 training (in a venv or conda env):
pip install torch torchaudio  # match your CUDA/ROCm version
pip install accelerate transformers
pip install librosa soundfile pyyaml tensorboard
pip install munch phonemizer huggingface_hub
pip install Cython  # required to build monotonic_align
```

---

## Step 1: Dataset Preparation

StyleTTS2's dataloader (`meldataset.py`) expects a file list in the format:

```
path/to/audio.wav|IPA phoneme string|speaker_name
```

Where:
- `path/to/audio.wav` is relative to `root_path` in the config
- The IPA string uses Kokoro's 178-token vocabulary (see [Phoneme Compatibility Reference](#phoneme-compatibility-reference))
- `speaker_name` is arbitrary for single-speaker setups

### Audio Requirements

- **Format:** WAV, mono
- **Sample rate:** 24000 Hz
- **Bit depth:** 16-bit
- **Duration:** 2–30 seconds per clip (the `prepare_dataset.py` filter enforces this range). Longer clips preserve natural sentence-level prosody and phrasing, which is valuable for Stage 2 training. The hard upper limit comes from PLBERT's 510-token `max_position_embeddings` — at typical German speaking rates, ~30 seconds of speech stays within this. Note that longer clips increase VRAM usage since the collater pads all batch items to the longest mel in the batch.
- **Min phoneme length:** 50 chars (configurable via `min_length` in config)

If your source audio is MP3, convert with ffmpeg:

```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 output.wav
```

### IPA Phonemization

Use `misaki` with `espeak-ng` backend:

```python
from misaki import espeak
g2p = espeak.EspeakG2P(language='de')  # or 'en-us', 'fr', etc.
phonemes, _ = g2p(text)
```

**Critical: `ʏ` is not in Kokoro's vocabulary.** The short-ü sound (`ʏ`, U+028F) is absent from Kokoro's 178-token set. Map it to `y` (U+0079, long ü) in post-processing — the duration distinction is learned from the audio:

```python
phonemes = phonemes.replace('ʏ', 'y')
```

Verify all output characters appear in the Kokoro symbol list before writing the file list. Any unknown characters will be silently dropped by `TextCleaner`, which can corrupt phoneme sequences.

### Train/Val Split

A 95/5 split works well. Keep the validation set small enough that the per-epoch validation loop finishes in a few minutes.

### OOD Texts

StyleTTS2's training code references an OOD (out-of-distribution) texts file (`OOD_data` in config). This is a plain text file with one IPA phoneme sequence per line, each at least 50 characters long. It is used for TensorBoard audio samples during training. 20 diverse sentences is sufficient.

---

## Step 2: Weight Conversion

Kokoro's weights are published on HuggingFace in a format specific to `KModel`. StyleTTS2's `load_checkpoint` expects a different structure. The conversion:

1. Download the Kokoro-82M checkpoint from HuggingFace
2. Load it with `torch.load`
3. Strip `module.` prefixes from all keys (present if saved from DataParallel)
4. Wrap in `{'net': {'bert': ..., 'bert_encoder': ..., 'predictor': ..., 'text_encoder': ..., 'decoder': ...}}`
5. Save as a new `.pth` file

```python
import torch

raw = torch.load('kokoro-v1_0.pth', weights_only=False)

# Strip module. prefix if present
def strip_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

net = {
    'bert': strip_prefix(raw['bert']),
    'bert_encoder': strip_prefix(raw['bert_encoder']),
    'predictor': strip_prefix(raw['predictor']),
    'text_encoder': strip_prefix(raw['text_encoder']),
    'decoder': strip_prefix(raw['decoder']),
}
torch.save({'net': net}, 'kokoro_base.pth')
```

Set `load_only_params: true` in the config so StyleTTS2 uses `strict=False` when loading — this silently ignores missing keys for components not present in Kokoro (diffusion network, SLM discriminator).

### Architecture Reference

Kokoro-82M component shapes (for verifying successful weight loading):

| Component | Parameters |
|-----------|-----------|
| bert (PLBERT) | 6.29M, 12 layers, hidden=768, vocab=178 |
| bert_encoder | 0.39M, Linear(768 → 512) |
| predictor | 16.19M, style_dim=128 |
| text_encoder | 5.61M, embedding(178, 512) |
| decoder | 53.28M, ISTFTNet, upsample=[10,6] |
| **Total** | **81.76M** |

Voicepack shape: `[510, 1, 256]` per voice (float32). First 128 dims → decoder, last 128 dims → predictor.

---

## Step 3: Symbol Mapping (Critical)

This is the single most important thing to get right. **Skipping or getting this wrong silently scrambles the pre-trained embeddings** — training will appear to proceed normally but the model will produce garbage.

### The Problem

Kokoro-82M and the default StyleTTS2 code both use 178 IPA tokens, but with **different index assignments**. For example:

| Index | StyleTTS2 default | Kokoro-82M |
|-------|-------------------|------------|
| 12 | `«` | `(` |
| 13 | `»` | `)` |
| 17 | `A` (uppercase) | `̃` (combining tilde) |
| 18 | `B` | `ʣ` |
| 19 | `C` | `ʥ` |
| 20 | `D` | `ʦ` (ts affricate) |
| 24 | `H` | `A` (uppercase starts here in Kokoro) |

If you use StyleTTS2's default symbol list, the token embedding layer will treat the German affricate `ʦ` as if it were the letter `D`, etc. The model loads without error but every phoneme is misinterpreted.

### The Fix

Generate `kokoro_symbols.py` from Kokoro's `config.json` (the `vocab` field). Place it in the `StyleTTS2/` directory. Then replace `StyleTTS2/text_utils.py` entirely with:

```python
# Kokoro-82M symbol mapping (replaces default StyleTTS2 symbols)
from kokoro_symbols import symbols, dicts, TextCleaner
```

The `kokoro_symbols.py` file must export:
- `symbols`: list of 178 strings in index order
- `dicts`: dict mapping symbol string → index
- `TextCleaner`: class with `__call__(text) -> list[int]` that maps each character to its index, silently skipping unknowns

Verify it works:

```bash
cd StyleTTS2
python -c "
from kokoro_symbols import symbols, dicts, TextCleaner
assert len(symbols) == 178
tc = TextCleaner()
ids = tc('bˈyːçɜ')
print(f'OK: {len(symbols)} symbols, sample IDs: {ids}')
"
```

---

## Step 4: StyleTTS2 Environment Setup

### Clone with Submodule (Recommended)

This repository includes a **patched fork of StyleTTS2** as a git submodule with all the critical fixes applied:

```bash
git clone --recurse-submodules https://github.com/semidark/kokoro-deutsch
cd kokoro-deutsch/StyleTTS2
```

The submodule (`semidark/StyleTTS2`) contains all necessary patches:
- ✅ weight_norm API migration (new parametrizations API)
- ✅ DataParallel wrapping order fix
- ✅ Missing adversarial logic restored
- ✅ GAN discriminator activation fix
- ✅ Diffusion sampler bypass
- ✅ F0 tensor shape bug fix
- ✅ Checkpoint save order fix
- ✅ PLBERT max sequence filter
- ✅ PyTorch 2.6+ weights_only compatibility

### Using Upstream StyleTTS2 (Not Recommended)

If you must use the upstream repository, you will need to apply **all** the patches manually. See the [Known Issues & Fixes](#known-issues--fixes-in-styletts2) section for the complete list of required changes.

```bash
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
# Then manually apply all patches from the sections below
```

### Download Required Utility Models

Three utility models are required by `train_first.py`. All three must be present even if their corresponding losses are disabled in the config — the code fails to initialize without them.

```bash
# JDC Pitch Extractor (required for F0 loss)
mkdir -p Utils/JDC
wget -O Utils/JDC/bst.t7 \
  https://github.com/yl4579/StyleTTS2/raw/main/Utils/JDC/bst.t7

# ASR Alignment Model (required for TMA alignment loss)
mkdir -p Utils/ASR
wget -O Utils/ASR/config.yml \
  https://github.com/yl4579/StyleTTS2/raw/main/Utils/ASR/config.yml
wget -O Utils/ASR/epoch_00080.pth \
  https://github.com/yl4579/StyleTTS2/raw/main/Utils/ASR/epoch_00080.pth

# PL-BERT (required — loaded even when diffusion is disabled)
# Download from the StyleTTS2 README "Pre-requisites" section
# Typically: huggingface-cli download yl4579/styletts2 Utils/PLBERT/ --local-dir .
mkdir -p Utils/PLBERT
# Place config.yml and the .pth checkpoint in Utils/PLBERT/
```

### Build monotonic_align

StyleTTS2 uses a Cython extension for fast monotonic alignment. It must be compiled from source:

```bash
cd StyleTTS2/monotonic_align
python setup.py build_ext --inplace
cd ..
```

This requires `Cython` and a C compiler (`gcc`/`clang`). On Ubuntu: `sudo apt-get install build-essential`. If this step fails, `import monotonic_align` will fail at training startup.

---

## Step 4.5: weight_norm API Migration (Critical)

This is the single most important compatibility step. **If you skip this, your trained checkpoints will NOT load into Kokoro's `KModel` for inference, and the `style_encoder` will produce infinite values in eval mode.**

### The Problem

StyleTTS2 uses PyTorch's **old** weight normalization API:
```python
from torch.nn.utils import weight_norm, spectral_norm  # OLD — deprecated since PyTorch 2.1
```

Kokoro's inference code (`KModel`) uses PyTorch's **new** parametrizations API:
```python
from torch.nn.utils.parametrizations import weight_norm, spectral_norm  # NEW
```

The two APIs produce **different state dict key formats**:
- Old: `conv1.weight_g`, `conv1.weight_v`
- New: `conv1.parametrizations.weight.original0`, `conv1.parametrizations.weight.original1`

If you train with the old API, the resulting checkpoint keys won't match what `KModel` expects. While PyTorch has backward compatibility for loading old keys into new modules, `KModel`'s loading logic has a brittle fallback path that corrupts keys under certain conditions.

Additionally, the old `spectral_norm` has a critical bug: it produces **different outputs in `.train()` vs `.eval()` mode**, causing the `StyleEncoder` to output infinite values in eval mode. The new API produces identical results in both modes.

### The Fix

Change **5 import lines** in StyleTTS2:

**`models.py` line 13:**
```python
# FROM:
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
# TO:
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
```

**`Modules/istftnet.py` line 5:**
```python
# FROM:
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
# TO:
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
```
Also replace all `remove_weight_norm(layer)` calls with `remove_parametrizations(layer, 'weight')`.

**`Modules/hifigan.py` line 5:** Same changes as `istftnet.py`.

**`Modules/discriminators.py` line 5:**
```python
# FROM:
from torch.nn.utils import weight_norm, spectral_norm
# TO:
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
```

No other code changes are needed. The `weight_norm()` and `spectral_norm()` function calls are identical between old and new APIs -- only the import path and internal storage format differ.

### Why This Works

- PyTorch's new `parametrizations.weight_norm` is a drop-in replacement for the old API at the call site level
- The old-format `kokoro_base.pth` checkpoint (with `weight_g`/`weight_v` keys) loads into new-API models via PyTorch's backward compatibility through `strict=False`
- After training, checkpoints contain new-format keys that `KModel` can load natively
- The new `spectral_norm` produces identical results in `.train()` and `.eval()` mode, eliminating the infinite-output bug

### Verification

After making the changes, verify:

```bash
cd StyleTTS2
python -c "
from models import build_model
# Should import without errors or deprecation warnings about weight_norm
print('Import OK')
"
```

---

## Step 5: Config File

The config lives outside StyleTTS2 (e.g., `configs/config_german_ft.yml`) and is passed via `--config_path`. The training command is always run **from inside the `StyleTTS2/` directory**, so all relative paths in the config need the `../` prefix.

### Critical: Top-Level vs Nested Parameters

`train_first.py` reads several critical parameters from the **top level** of the YAML, not from the nested `training:` block:

```python
# From train_first.py — these read from the TOP LEVEL:
batch_size = config.get("batch_size", 10)
epochs = config.get("epochs_1st", 200)
saving_epoch = config.get("save_freq", 2)
pretrained_model = config["pretrained_model"]
load_only_params = config.get("load_only_params", True)
```

If you put these only inside `training:`, they will be silently ignored and defaults will be used. Always set them at the top level:

```yaml
# TOP LEVEL — these are actually read by train_first.py
batch_size: 4
epochs_1st: 10
save_freq: 1          # save every epoch; default is 2 (skips odd epochs)
pretrained_model: "../training/kokoro_base.pth"
load_only_params: true
```

### Full Working Config

```yaml
# Top-level params read directly by train_first.py
batch_size: 4
epochs: 10
epochs_1st: 10
save_freq: 1
pretrained_model: "../training/kokoro_base.pth"
load_only_params: true

log_dir: "logs/kokoro_german"

data_params:
  train_data: "../training/train_list.txt"
  val_data: "../training/val_list.txt"
  root_path: "../dataset/audio"
  OOD_data: "../training/OOD_texts.txt"
  min_length: 50

preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300
    n_mels: 80
    fmin: 0
    fmax: 8000

model_params:
  dim_in: 64
  n_token: 178
  hidden_dim: 512
  style_dim: 128
  max_dur: 50
  multispeaker: false
  n_mels: 80
  dropout: 0.2
  n_layer: 3
  text_encoder_kernel_size: 5

  decoder:
    type: istftnet          # Kokoro uses ISTFTNet, NOT hifigan
    upsample_rates: [10, 6]
    upsample_kernel_sizes: [20, 12]
    upsample_initial_channel: 512
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    gen_istft_n_fft: 20
    gen_istft_hop_size: 5

  diffusion:                # Required in config even though disabled
    embedding_mask_proba: 0.1
    transformer:
      num_layers: 3
      num_heads: 8
      head_features: 64
      multiplier: 2
    dist:
      sigma_data: 0.2
      estimate_sigma_data: true
      mean: -3.0
      std: 1.0

  plbert:
    hidden_size: 768
    num_attention_heads: 12
    intermediate_size: 2048
    max_position_embeddings: 512
    num_hidden_layers: 12
    dropout: 0.1

  slm:
    model: "microsoft/wavlm-base-plus"
    sr: 16000
    hidden: 768
    nlayers: 13
    initial_channel: 64

loss_params:
  lambda_gen: 1.0
  lambda_mel: 5.0
  lambda_dur: 1.0
  lambda_ce: 20.0
  lambda_F0: 1.0
  lambda_norm: 1.0
  lambda_s2s: 1.0
  lambda_mono: 1.0
  lambda_slm: 1.0           # enable SLM adversarial loss for stage 2 (0 for stage 1)
  lambda_diff: 0.0          # disabled — no diffusion
  lambda_sty: 0.0
  TMA_epoch: 0              # start alignment immediately (fine-tuning, not scratch)
  diff_epoch: 999           # effectively disable diffusion training
  joint_epoch: 3            # start SLM/GAN adversarial training at epoch 3

optimizer_params:
  lr: 0.0001
  bert_lr: 0.00001          # lower LR for PLBERT
  ft_lr: 0.0001

F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00080.pth"
PLBERT_dir: "Utils/PLBERT/"

slmadv_params:
  min_len: 100
  max_len: 500
  batch_percentage: 0.5
  iter: 10
  thresh: 5
  scale: 0.01
  sig: 1.5
```

---

## Step 6: Smoke Testing

Before launching a multi-day training run, verify each component separately. This saves hours of debugging if something is wrong.

### 1. Verify symbol patch

```python
from kokoro_symbols import symbols, dicts, TextCleaner
assert len(symbols) == 178
tc = TextCleaner()
# Test a few German phonemes
assert dicts['ç'] == 78   # ich-Laut
assert dicts['ʦ'] == 20   # ts affricate
assert dicts['ː'] == 158  # length mark
print("Symbol patch OK")
```

### 2. Verify model loads

```python
import torch, yaml
from munch import Munch
from models import build_model

config = Munch(yaml.safe_load(open('../configs/config_german_ft.yml')))
model = build_model(config.model_params, text_aligner, pitch_extractor, plbert)
checkpoint = torch.load('../training/kokoro_base.pth', weights_only=False)
# Should print "Unexpected keys: ..." for diffusion/slm components — that's fine
# Should NOT print any "size mismatch" errors
```

### 3. Verify forward + backward pass

```python
# Run one batch through the model in train() mode
# Check that all losses are finite (not NaN or inf)
# A single step taking 60-120 seconds is normal on first run (kernel compilation)
```

### 4. Run 2 training steps

```bash
cd StyleTTS2
accelerate launch train_first.py --config_path ../configs/config_german_ft.yml
# Ctrl+C after you see "Step [2/...]" with non-NaN losses
```

**Healthy first-step losses (approximate):**

| Loss | Expected range |
|------|---------------|
| Mel Loss | 0.8–1.5 (drops fast) |
| Gen Loss | 3–6 |
| Disc Loss | 4–6 |
| Mono Loss | 0.01–0.1 |
| S2S Loss | 1–6 (drops over epochs) |
| SLM Loss | 1–3 |

Any NaN in Mel Loss is a red flag — most likely a symbol mapping problem causing the model to receive garbage token indices.

---

## Step 7: Training

### Launch command

Always run from the `StyleTTS2/` directory:

```bash
cd StyleTTS2
accelerate launch train_first.py --config_path ../configs/config_german_ft.yml
```

Logs go to `StyleTTS2/logs/kokoro_german/train.log`. Monitor with:

```bash
tail -f logs/kokoro_german/train.log
```

TensorBoard:

```bash
tensorboard --logdir StyleTTS2/logs/kokoro_german/tensorboard
```

### Checkpoints

With `save_freq: 1` at the top level, a checkpoint is saved after each epoch as:

```
logs/kokoro_german/epoch_1st_00000.pth   # epoch 0 (internal 0-indexed)
logs/kokoro_german/epoch_1st_00002.pth   # epoch 2
...
```

A final `first_stage.pth` is written when training completes all epochs.

### Resuming from a checkpoint

Update `pretrained_model` in the config to point to the latest checkpoint, and set `load_only_params: false` so the optimizer state and epoch counter are also restored:

```yaml
pretrained_model: "../StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth"
load_only_params: false
```

### Loss interpretation

| Loss | What it means | Healthy trend |
|------|--------------|---------------|
| Mel Loss | Mel spectrogram reconstruction | 0.8 → 0.25 over 10 epochs |
| Gen Loss | GAN generator vs discriminator | Stable 2.5–3.5 |
| Disc Loss | GAN discriminator | Stable 3.8–4.2 |
| Mono Loss | Monotonic alignment quality | < 0.05 — lower is better |
| S2S Loss | Sequence-to-sequence alignment | Declining over time |
| SLM Loss | WavLM feature matching | Stable or slowly declining |

If Mel Loss plateaus above 0.4 after several epochs, something is wrong (data quality, phoneme mapping, or learning rate).

### Timing

On an AMD Radeon 8060S (Strix Halo) with ROCm 7.12, fp32, batch_size=4:

- ~105 seconds per step
- ~2743 steps per epoch (11,551 clips / batch_size=4 / ~1.05 clips per step)
- ~4.8 hours per epoch
- ~48 hours for 10 epochs

On an NVIDIA A100 with batch_size=16 you would expect ~2–4 hours total for 10 epochs.

---

## Step 7.5: Stage 2 Training (Prosody Predictor)

Stage 1 trains the decoder (mel reconstruction) but does NOT train the prosody predictor for the new language. Without Stage 2, the model will produce speech with English rhythm, stress, and intonation patterns.

Stage 2 (`train_second.py`) trains the `predictor`, `bert_encoder`, `bert`, and `predictor_encoder` components to predict German-appropriate duration, F0 (pitch), and energy patterns from text.

### Config for Stage 2

Add `epochs_2nd` to the top level of the config (read by `train_second.py`):

```yaml
epochs_2nd: 10
first_stage_path: "first_stage.pth"  # Stage 1 checkpoint to load (relative to log_dir)
```

**Important config changes for Stage 2:**

The default config values have been updated to work correctly with the fixes:

```yaml
# In loss_params section:
joint_epoch: 3                    # Start adversarial training at epoch 3 (not 999)
lambda_slm: 1.0                   # Enable SLM adversarial loss

# In training section:
second_stage_load_pretrained: false  # false = load from first_stage.pth (recommended)
                                      # true = load from kokoro_base.pth (not recommended)
```

**Why `second_stage_load_pretrained: false`?**
- `false` loads from your trained `first_stage.pth`, which has a properly trained `style_encoder`
- `true` loads from `kokoro_base.pth`, which requires `strict=False` and can silently fail to load certain modules

**Joint training settings:**
- `joint_epoch: 3` starts the GAN discriminator and SLM adversarial losses at epoch 3
- This provides gradient signal to stabilize the `style_encoder` and improve prosody quality
- Earlier values (like `joint_epoch: 1`) work but may be unstable initially
- The original `joint_epoch: 999` effectively disabled these losses, contributing to the style encoder collapse

Stage 2 loads the Stage 1 checkpoint automatically. By default it looks for `logs/kokoro_german/first_stage.pth` (written when Stage 1 completes all epochs). If you stopped Stage 1 early, set `first_stage_path` to the specific checkpoint:

```yaml
first_stage_path: "epoch_1st_00003.pth"   # relative to log_dir
```

### Launch

```bash
cd StyleTTS2
accelerate launch train_second.py --config_path ../configs/config_german_ft.yml
```

### Stage 2 loss interpretation

| Loss | What it means | Healthy trend |
|------|--------------|---------------|
| Loss (Mel) | Mel reconstruction with predicted prosody | **~0.43** at start, declining to ~0.25 |
| Dur Loss | Duration prediction accuracy | 1.3 → 0.9 over 10 epochs |
| CE Loss | Alignment cross-entropy | 0.18 → 0.05 |
| Norm Loss | Energy (N) prediction | 3.0 → 0.8 (noisy) |
| F0 Loss | Pitch contour prediction | 4.1 → 1.8 over 10 epochs |
| LM Loss | WavLM feature matching | Stable ~2.95 |
| Gen/Disc Loss | GAN discriminator losses | Activate at `joint_epoch`, stable ~2-4 |
| SLM Loss | WavLM adversarial loss | Activate at `joint_epoch`, stable ~1-3 |

**Important:** With the checkpoint loading fixes, Stage 2 Mel loss now starts at **~0.43** (indicating pretrained weights loaded correctly) rather than **~7.5-8.0** (indicating random initialization). 

If you see Mel loss starting above 2.0, check that:
1. `pretrained_model` points to your Stage 1 checkpoint
2. `load_only_params: true` is set
3. You're using the patched `train_second.py` from the submodule (DataParallel after loading)

The Mel loss is higher than Stage 1 because the decoder receives *predicted* F0/energy instead of ground truth, but with proper weight loading it should start around 0.4, not 7.5.

### Critical Finding: Stage 2 Style Encoder Degradation (RESOLVED)

**Status:** ✅ **FIXED** — Stage 2 training now works correctly. The issues below were caused by bugs that have been resolved.

**Previous Symptoms:**
- Stage 1 training succeeded, producing intelligible (if robotically-timed) speech.
- Stage 2 would complete training but output pure static noise.
- The extracted voicepack had an exceptionally low norm (~0.38) or exploded to `NaN`.

**Root Causes (Now Fixed):**

1. **DataParallel Bug:** `load_checkpoint()` was being called *after* models were wrapped in `MyDataParallel` (which prepends `module.` to all keys). Because `strict=False` was used, checkpoint keys silently failed to match, loading zero weights. The `style_encoder` initialized randomly, its `spectral_norm` collapsed, and the output exploded to ~1e17, causing static noise.
   - **Fix:** Moved `MyDataParallel` wrapping to *after* checkpoint loading in `train_second.py`.

2. **Discarded Pretrained Weights:** The `ignore_modules` list in `train_second.py` was actively discarding `bert`, `bert_encoder`, and `predictor` from the Stage 1 checkpoint, forcing the model to throw away pretrained English Kokoro prosody knowledge.
   - **Fix:** Removed these modules from `ignore_modules`.

3. **Missing Adversarial Logic:** The code computing `y_rec_gt` and `y_rec_gt_pred` had been accidentally deleted, making it impossible to enable `joint_epoch` without crashing.
   - **Fix:** Restored the missing ground-truth logic.

4. **GAN Discriminator Inactive:** The GAN discriminator activation (`start_ds`) was gated on `diff_epoch`. Since Kokoro disables diffusion (`diff_epoch=999`), the GAN discriminator never turned on.
   - **Fix:** Gated `start_ds` on `joint_epoch` instead.

5. **Diffusion Sampler Garbage:** The `SLMAdversarialLoss` was sampling style embeddings from the diffusion model even when disabled, feeding garbage to the discriminator.
   - **Fix:** Added a `diffusion_enabled` flag to bypass the sampler.

**Current Behavior:**
- Stage 2 now correctly loads pretrained weights from Stage 1
- Mel loss starts at ~0.43 (indicating trained weights) instead of ~8.33 (random weights)
- Adversarial training activates correctly at `joint_epoch`
- TensorBoard shows Kokoro-faithful inference audio using predicted duration/pitch

### Kokoro-Faithful TensorBoard Audio

The training code now generates **Kokoro-faithful audio samples** in TensorBoard at the end of each epoch. Instead of using ground-truth F0/energy (which produces misleadingly good results), the samples are generated using the actual inference path:

1. Extract a mini voicepack from the current model's style encoder
2. Predict duration, F0, and energy using the `predictor`
3. Decode through the `decoder`
4. Compare against Stage 1 baseline at epoch 0

This gives you accurate previews of how the model will sound when used with `KPipeline` inference. The audio is saved to TensorBoard at `train/epoch_X/audio` for each of 7 test sentences.

### Checkpoint conversion for inference

After Stage 2, convert the checkpoint for use with Kokoro's `KModel`:

```bash
python scripts/test_inference.py \
    --checkpoint StyleTTS2/logs/kokoro_german/epoch_2nd_00009.pth \
    --voicepack voices/dm_daniel.pt \
    --output-dir test_output/
```

The conversion extracts the 5 inference components (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`) and ensures the `module.` prefix is present for KModel's loading path. All components are fully fine-tuned for German since the weight_norm API migration ensures format compatibility.

---

## Step 8: Voicepack Extraction

After training, you need a `.pt` voicepack file (shape `[510, 1, 256]`) for inference with `KPipeline`.

### Strategy 1: Reuse an existing voicepack

Try this first. The fine-tuned model weights encode the language capability; the voicepack controls voice timbre and prosody style. Load an existing English male voicepack (`am_adam.pt`) and test with the fine-tuned model:

```python
from kokoro import KPipeline
pipeline = KPipeline(lang_code='d', repo_id='path/to/your/model')
audio_gen = pipeline("Hallo Welt", voice='am_adam')
```

This works surprisingly well if the voice character doesn't need to match the training speaker exactly.

### Strategy 2: Style encoder averaging

Extract the voicepack by running the fine-tuned model's style encoder on representative utterances using the `extract_voicepack.py` script.

If your Stage 2 `style_encoder` has degraded or collapsed (as described in the Critical Finding above), you can use the `--style-encoder-model` flag to extract the acoustic style vector from your Stage 1 checkpoint, while retaining the predictor capabilities of your Stage 2 checkpoint:

```bash
python scripts/extract_voicepack.py \
    --model StyleTTS2/logs/kokoro_german/epoch_2nd_00009.pth \
    --style-encoder-model StyleTTS2/logs/kokoro_german/epoch_1st_00009.pth \
    --audio-dir path/to/training/audio \
    --output voices/dm_daniel.pt
```

For a manual approach in code:

```python
import torch
from models import load_ASR_models, load_F0_models, build_model

# Load fine-tuned model
# ... (load checkpoint)

model.eval()
voicepack_parts = []
with torch.no_grad():
    for wav_path in representative_clips:
        mel = compute_mel(wav_path)                    # [1, 80, T]
        style = model.style_encoder(mel.unsqueeze(1))  # [1, 256]
        voicepack_parts.append(style)

# Average and expand to voicepack shape [510, 1, 256]
mean_style = torch.stack(voicepack_parts).mean(0)     # [1, 256]
voicepack = mean_style.unsqueeze(0).expand(510, -1, -1)  # [510, 1, 256]
torch.save(voicepack, 'dm_daniel.pt')
```

### Strategy 3: Gradient optimization

Freeze model weights and optimize the voicepack tensor directly to minimize mel reconstruction loss on your training clips. Analogous to textual inversion for diffusion models. Takes ~30 minutes on GPU, produces the best match to the target speaker.

---

## AMD ROCm Notes

This section covers specifics for training on AMD GPUs via ROCm. This is not well-documented elsewhere.

### What works

- **ROCm 7.12** on AMD Radeon 8060S (Strix Halo / Ryzen AI Max) runs the full training loop successfully
- fp32 precision with batch_size=4 is fully stable
- PyTorch detects the GPU correctly; use `torch.cuda.is_available()` (ROCm maps to the CUDA API)

### What doesn't work

**fp16 / mixed precision:** Do not use. fp16 training causes silent crashes and hangs during MIOpen kernel compilation on this architecture. The training process will appear to stall or exit without an error message.

**Large batch sizes:** batch_size > 4 triggers slow or unstable MIOpen kernel tuning on first run and can silently hang. Stick with batch_size=4 for stability.

**accelerate config:** Use the simplest possible config — no mixed precision, no multi-GPU:

```yaml
# ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
mixed_precision: 'no'
num_processes: 1
```

Or generate it interactively:
```bash
accelerate config
# Choose: single machine, no distributed, no mixed precision
```

### MIOpen kernel compilation

On the very first training run, ROCm/MIOpen compiles GPU kernels for each operation. This can take 10–30 minutes before the first step completes. This is normal. Subsequent runs reuse the compiled kernels from cache (`~/.cache/miopen/`).

Do not kill the process during this phase — it will look like a hang but it isn't.

### Performance

With fp32 and batch_size=4 on an 8060S with 137GB unified memory:
- ~105 seconds per training step
- ~4.8 hours per epoch on 11,551 clips
- Full 10-epoch run: ~48 hours

This is slower than cloud NVIDIA but eliminates rental costs and allows unattended multi-day runs on your own hardware.

---

## Known Issues & Fixes in StyleTTS2

These bugs exist in the upstream StyleTTS2 repository as of early 2026. Both affect fine-tuning with Kokoro's weight format.

### Bug 1: F0 tensor shape mismatch at epoch boundaries

**File:** `train_first.py` line ~409, `train_second.py` line ~695

**Symptom:** Training crashes with this error at the end of each epoch (after validation completes):

```
RuntimeError: The size of tensor a (234600) must match the size of tensor b (9)
at non-singleton dimension 3
```

**Cause:** After validation, `train_first.py` generates TensorBoard audio samples in a per-sample loop. Inside this loop, `gt` already has a batch dimension (`unsqueeze(0)` on line ~405), so the pitch extractor returns `F0_real` with shape `(1, T)`. A spurious extra `F0_real = F0_real.unsqueeze(0)` then makes it `(1, 1, T)`. This propagates as a 4D tensor through `Decoder` → `Generator` → `SineGen`, where the harmonic multiplier of shape `(1, 1, 9)` cannot broadcast against dimension 3 of size `T*300`.

The number 234600 is `mel_length * 300` (300 = upsample factor), confirming the raw waveform-length dimension is leaking into the wrong axis.

**Fix:** Remove the spurious `unsqueeze(0)` line:

```python
# In train_first.py and train_second.py, find and remove this line:
F0_real = F0_real.unsqueeze(0)   # DELETE THIS LINE
```

The surrounding context in `train_first.py`:
```python
# BEFORE (buggy):
F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
F0_real = F0_real.unsqueeze(0)    # <-- remove this
s = model.style_encoder(gt.unsqueeze(1))

# AFTER (fixed):
F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
s = model.style_encoder(gt.unsqueeze(1))
```

### Bug 2: Checkpoint saved after TensorBoard audio generation

**File:** `train_first.py` lines ~523–535

**Symptom:** If the TensorBoard audio generation crashes (e.g., Bug 1 above), the epoch's checkpoint is never saved. The entire epoch's training is lost.

**Cause:** In the original code, the order inside `if accelerator.is_main_process:` is:
1. Log validation metrics
2. Generate TensorBoard audio samples  ← crash here loses the checkpoint
3. Save checkpoint

**Fix:** Move the checkpoint save block to before the TensorBoard audio generation:

```python
# Correct order:
if accelerator.is_main_process:
    # 1. Log metrics
    writer.add_scalar(...)
    writer.add_figure(...)

    # 2. Save checkpoint FIRST
    if epoch % saving_epoch == 0:
        state = { ... }
        torch.save(state, save_path)

    # 3. Generate TensorBoard audio (crash here no longer loses the checkpoint)
    with torch.no_grad():
        for bib in range(len(asr)):
            ...
```

### Bug 3: `train_second.py` missing train() mode after checkpoint load

**File:** `train_second.py` lines ~313–319

**Symptom:** All losses are NaN from step 1 of Stage 2.

**Cause:** After `load_checkpoint()` (which sets all modules to `.eval()` mode), `train_second.py` only sets specific modules back to `.train()` mode:
```python
model.predictor.train()
model.bert_encoder.train()
model.bert.train()
```
It forgets `style_encoder`, `predictor_encoder`, `decoder`, and `text_encoder`. With the old `spectral_norm` API, the `style_encoder` produces infinite values in eval mode, which propagate through the predictor and decoder as NaN.

**Fix:** Replace the selective `.train()` calls with:
```python
_ = [model[key].train() for key in model]
```

This is required even after migrating to the new `spectral_norm` API, as a defensive measure.

### Bug 4: Hardcoded ipdb breakpoint in `train_second.py`

**File:** `train_second.py` line ~533

**Symptom:** Training hangs after the first backward pass in Stage 2, dropping into an interactive debugger that can't receive input under `accelerate launch`.

**Fix:** Remove the `from IPython.core.debugger import set_trace` and `set_trace()` lines.

### Bug 5: PLBERT max sequence length overflow

**File:** `meldataset.py` (dataloader)

**Symptom:** `RuntimeError: The expanded size of the tensor (527) must match the existing size (512)` -- crashes when a phoneme sequence exceeds PLBERT's 512-token `max_position_embeddings`.

**Fix:** Filter the training data to exclude sequences longer than 510 tokens:
```python
self.data_list = [d for d in self.data_list if len(self.text_cleaner(d[1])) <= 510]
```

### Bug 6: `torch.load` defaults to `weights_only=True` in PyTorch 2.6+

**File:** `models.py` `load_checkpoint`, various `torch.load` calls

**Symptom:** `_pickle.UnpicklingError: Weights only load failed` when loading old-format checkpoints.

**Fix:** Add a monkey-patch at the top of `train_first.py` and `train_second.py`, before any model imports:
```python
import torch
if getattr(torch, "_original_load", None) is None:
    torch._original_load = torch.load
    torch.load = lambda *args, **kwargs: torch._original_load(
        *args, **{**kwargs, "weights_only": False}
    )
```

### Bug 7: DataParallel wrapping before checkpoint loading (CRITICAL)

**File:** `train_second.py` lines ~280–320

**Symptom:** Stage 2 training appears to work but produces static noise during inference. Mel loss starts at ~7.5–8.0 instead of ~0.4. Voicepack norm is extremely low (~0.38) or explodes to `NaN`.

**Cause:** The model was wrapped in `MyDataParallel` **before** `load_checkpoint()` was called. `MyDataParallel` prepends `module.` to all state dict keys. Because `load_checkpoint` uses `strict=False`, the mismatching keys (with `module.` prefix in the model but without in the checkpoint) were silently ignored. The `style_encoder` started from random initialization, causing spectral_norm to collapse and output to explode to ~1e17.

**Fix:** Move the `MyDataParallel` wrapping to **after** checkpoint loading:

```python
# BEFORE (buggy):
model = MyDataParallel(...)  # model now has module. prefix
model, optimizer, ... = load_checkpoint(...)  # checkpoint keys don't match

# AFTER (fixed):
model, optimizer, ... = load_checkpoint(...)  # load first with matching keys
model = MyDataParallel(...)  # wrap after loading
```

This fix is in the `semidark/StyleTTS2` submodule.

### Bug 8: `ignore_modules` discards pretrained weights (CRITICAL)

**File:** `train_second.py` `load_checkpoint` call

**Symptom:** Stage 2 prosody predictor starts from random initialization instead of pretrained English weights. Duration and pitch predictions are poor quality even after training.

**Cause:** The `ignore_modules` list included `bert`, `bert_encoder`, and `predictor`, causing these modules to be excluded from checkpoint loading. Stage 2 was forced to train these from scratch instead of fine-tuning the pretrained English prosody knowledge.

**Fix:** Remove `bert`, `bert_encoder`, `predictor` from `ignore_modules`. The complete Stage 2 loading should be:

```python
model, optimizer, ... = load_checkpoint(
    model, 
    optimizer, 
    ...
    # ignore_modules=['decoder', 'style_encoder']  # Only these if needed
)
```

The patched `train_second.py` in the submodule no longer excludes these critical modules.

### Bug 9: Missing adversarial ground-truth computation

**File:** `train_second.py` training loop

**Symptom:** Enabling `joint_epoch` (adversarial training) causes `NameError: name 'y_rec_gt' is not defined` or crashes with missing tensors.

**Cause:** The code computing `y_rec_gt` and `y_rec_gt_pred` (required for WavLM adversarial loss) was accidentally deleted in a previous cleanup.

**Fix:** Restore the ground-truth computation in the training loop before the adversarial loss section.

### Bug 10: GAN discriminator gated on wrong epoch

**File:** `train_second.py` discriminator activation

**Symptom:** GAN discriminator never activates even when `joint_epoch` is set. Only SLM adversarial loss runs, no GAN losses.

**Cause:** The discriminator activation was gated on `diff_epoch` instead of `joint_epoch`. Since Kokoro sets `diff_epoch=999` (disabling diffusion), the discriminator never turned on.

**Fix:** Change the gating condition from `diff_epoch` to `joint_epoch`.

### Bug 11: Diffusion sampler runs when disabled

**File:** `Modules/slmadv.py` `SLMAdversarialLoss`

**Symptom:** SLM adversarial loss produces unstable/nonsensical values when diffusion is disabled. Style embeddings appear random.

**Cause:** The `SLMAdversarialLoss` tried to sample style embeddings from the diffusion model even when `diff_epoch=999` (diffusion disabled), feeding garbage to the discriminator.

**Fix:** Add a `diffusion_enabled` flag that bypasses the diffusion sampler and uses ground-truth style vectors directly when diffusion is disabled.

---

## Phoneme Compatibility Reference

### German IPA symbols in Kokoro's vocabulary

All standard German phonemes are covered by Kokoro's 178-token set:

| Sound | IPA | Unicode | Kokoro ID |
|-------|-----|---------|-----------|
| ich-Laut | `ç` | U+00E7 | 78 |
| ach-Laut | `x` | U+0078 | 66 |
| ö long | `ø` | U+00F8 | 116 |
| ö short | `œ` | U+0153 | 120 |
| ü long | `y` | U+0079 | 67 |
| ts affricate | `ʦ` | U+02A6 | 20 |
| schwa-r | `ɐ` | U+0250 | 70 |
| sch | `ʃ` | U+0283 | 131 |
| ng | `ŋ` | U+014B | 112 |
| vowel length | `ː` | U+02D0 | 158 |
| schwa | `ə` | U+0259 | 83 |
| uvular r | `ʁ` | U+0281 | 128 |
| glottal stop | `ʔ` | U+0294 | 148 |

### Missing symbol and fix

| IPA | Unicode | Meaning | Fix |
|-----|---------|---------|-----|
| `ʏ` | U+028F | short ü | Map to `y` (U+0079) |

`ʏ` is produced by `espeak-ng` for short ü (e.g., in "Bücher"). It is not in Kokoro's vocabulary. Replace it with `y` (long ü) in post-processing. The model learns the duration difference from the audio context.

### Diacritics (stress markers)

| Symbol | Meaning | Kokoro ID |
|--------|---------|-----------|
| `ˈ` | primary stress | 156 |
| `ˌ` | secondary stress | 157 |

These are produced by `espeak-ng` and are in Kokoro's vocabulary. Do not strip them.
