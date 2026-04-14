# AGENTS.md — kokoro-deutsch

## Project Overview

kokoro-deutsch is a training recipe for fine-tuning [Kokoro TTS](https://github.com/hexgrad/kokoro) (82M parameters, based on StyleTTS 2) for German. The project contains:

- A lightly modified copy of the `kokoro/` inference package (German language code added)
- A patched fork of `StyleTTS2/` as a git submodule (`semidark/StyleTTS2`, branch `main`)
- Original scripts for dataset preparation, voicepack extraction, and inference testing
- Training and troubleshooting docs split by purpose

- **Primary language:** Python 3.10–3.13
- **Package manager:** `uv` (lockfile: `uv.lock`)
- **Build backend:** hatchling
- **License:** Apache 2.0
- **Repository:** `https://github.com/semidark/kokoro-deutsch`

## Build & Install

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/semidark/kokoro-deutsch
cd kokoro-deutsch

# Install Python dependencies (use uv, not pip directly)
uv sync

# Install the package in editable mode
uv pip install -e .

# System dependency required for German G2P
# macOS: brew install espeak-ng
# Linux: apt-get install espeak-ng
```

## Running Tests

No repository-level test suite is currently maintained in this repo.

To validate training/inference changes, use:

- Smoke training runs in `StyleTTS2/`
- `scripts/test_inference.py` for checkpoint conversion + inference checks

## CLI Usage

```bash
# Generate speech from text
uv run kokoro --text "Hallo Welt" -o output.wav -l d --voice dm_daniel

# From file
uv run kokoro -i input.txt -o output.wav -l d --voice dm_daniel
```

## Code Style Guidelines

### Python

No linter or formatter is configured. Follow the existing conventions observed in the codebase:

#### Imports

- **Order:** Relative imports first, then third-party, then stdlib — but the codebase is inconsistent. Prefer: relative imports, third-party (alphabetical), stdlib (alphabetical).
- **Style:** Use `from x import Y` for specific items; use `import x` for modules used with qualification (e.g., `import torch`, `import re`).
- Common aliases: `import torch.nn as nn`, `import torch.nn.functional as F`, `import numpy as np`.
- Conditional imports are used for optional dependencies (e.g., `misaki[ja]`, `misaki[zh]`) — wrap in try/except with `logger.error()` before re-raising.
- Use `TYPE_CHECKING` guard for imports only needed for type hints (see `__main__.py`).

#### Formatting

- **Quotes:** Single quotes for strings (e.g., `'ab'`, `'cpu'`). Double quotes acceptable in docstrings.
- **Line length:** No hard limit configured; lines up to ~150 chars exist. Keep reasonable.
- **Indentation:** 4 spaces (standard Python).
- **Trailing commas:** Used in multiline constructs (function args, lists).
- **Blank lines:** Two blank lines between top-level classes/functions, one blank line between methods.

#### Types & Type Hints

- Use `typing` module imports: `Optional`, `Union`, `List`, `Tuple`, `Generator`, `Callable`.
- Modern Python syntax (`tuple[X, Y]` lowercase) appears in `model.py` — both styles coexist.
- Annotate public method signatures. Internal/private methods may omit hints.
- Use `@dataclass` for simple data containers (see `KPipeline.Result`, `KModel.Output`).

#### Naming Conventions

- **Classes:** PascalCase with `K` prefix for public API (`KModel`, `KPipeline`). Neural network modules use standard PyTorch naming (`CustomAlbert`, `ProsodyPredictor`, `TextEncoder`).
- **Functions/methods:** snake_case (`load_voice`, `forward_with_tokens`).
- **Constants:** UPPER_SNAKE_CASE (`ALIASES`, `LANG_CODES`, `MODEL_NAMES`, `MAGIC_DIVISOR`).
- **Private methods:** Prefixed with underscore (`_f02uv`, `_shortcut`, `_residual`, `_build_weights`).
- **Variables:** Short names acceptable in math-heavy code (`x`, `s`, `d`, `F0`, `N`). Use descriptive names elsewhere.
- **Files:** snake_case (`custom_stft.py`, `istftnet.py`).

#### Error Handling

- Use `assert` for internal invariants and input validation (e.g., `assert lang_code in LANG_CODES`).
- Raise `ValueError` for invalid user inputs with descriptive messages.
- Raise `RuntimeError` for environment issues (CUDA/MPS unavailability).
- Use `try/except` with `logger.error()` for optional dependency failures, then re-raise.
- Bare `except:` exists in `model.py:73` for state_dict loading fallback — avoid this pattern in new code; catch specific exceptions.

#### Logging

- Use `loguru` logger (imported from `loguru import logger`), NOT `print()` for diagnostics.
- `print()` is acceptable in `scripts/` CLI tools for direct user output.
- Log levels: `logger.debug()` for tracing, `logger.warning()` for non-fatal issues, `logger.error()` for failures.
- Logger is disabled by default in `__init__.py` (`logger.disable("kokoro")`).

#### Classes & Architecture

- **KModel** (`model.py`): `torch.nn.Module` subclass. Language-blind. Handles weight loading and forward pass. Use `@torch.no_grad()` for inference methods.
- **KPipeline** (`pipeline.py`): Language-aware orchestrator. Handles G2P, voice loading, chunking, and inference. Designed as one instance per language; reuse `KModel` across pipelines.
- Inner classes/dataclasses are nested (e.g., `KPipeline.Result`, `KModel.Output`).
- Use `@staticmethod` for methods that don't access instance state.
- Use `@property` for computed attributes.

#### Docstrings

- Triple-single-quote (`'''`) block docstrings on classes (see `KPipeline`, `KModel`).
- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections on some methods.
- Inline comments for non-obvious logic (e.g., STFT math, timestamp calculations).

## Project Structure

```
kokoro/              # Inference package (from hexgrad/kokoro, German lang code added)
  __init__.py        # Version, logging setup, public exports (KModel, KPipeline)
  __main__.py        # CLI entry point
  pipeline.py        # KPipeline: G2P + voice management + inference orchestration
  model.py           # KModel: neural network, weight loading, forward pass
  modules.py         # CustomAlbert, ProsodyPredictor, TextEncoder
  istftnet.py        # ISTFT-based vocoder (Decoder, Generator, TorchSTFT)
  custom_stft.py     # ONNX-compatible STFT (no unfold/complex ops)
StyleTTS2/           # Training code (git submodule: semidark/StyleTTS2, branch main)
scripts/             # Original: dataset prep, voicepack extraction, inference testing
  prepare_dataset.py # Audio processing + IPA phonemization pipeline
  prepare_training.py# Train/val split, mel/F0 precomputation, weight conversion
  extract_voicepack.py # Extract .pt voicepack from trained checkpoint
  test_inference.py  # Convert checkpoint to KModel format + run test sentences
configs/             # Training configuration
  config_german_ft.yml # StyleTTS2 config for German fine-tuning
training/            # Training data lists, OOD texts, config (large files excluded)
docs/                # Documentation
  TRAINING_GUIDE.md  # Step-by-step training flow
  TROUBLESHOOTING.md # Debugging failures and fixes
  ARCHITECTURE.md    # Technical compatibility reference
  images/            # TensorBoard screenshots
```

## Key Technical Details

- **Sample rate:** 24000 Hz
- **Max phoneme length:** 510 characters (inputs are chunked/truncated to fit)
- **German G2P:** Uses `espeak.EspeakG2P(language='de')` from misaki, requires espeak-ng installed
- **Voice files:** `.pt` files; shape `[510, 1, 256]` per voice (float32)
- **Device selection:** Auto-detects CUDA > MPS > CPU; explicit device can be passed
- **ONNX export:** Use `disable_complex=True` in `KModel` to use `CustomSTFT` instead of `TorchSTFT`

## Training Pipeline

Training uses the patched [StyleTTS2](https://github.com/yl4579/StyleTTS2) code in the `StyleTTS2/` submodule. See `docs/TRAINING_GUIDE.md` for the full guide.

### Critical: weight_norm API

StyleTTS2's modules have been migrated from `torch.nn.utils.weight_norm` (old, deprecated) to `torch.nn.utils.parametrizations.weight_norm` (new). This is **mandatory** for producing checkpoints compatible with Kokoro's `KModel` inference pipeline. The files affected are:

- `StyleTTS2/models.py`
- `StyleTTS2/Modules/istftnet.py`
- `StyleTTS2/Modules/hifigan.py`
- `StyleTTS2/Modules/discriminators.py`

Do NOT revert these imports. The old API produces state dict keys (`weight_g`/`weight_v`) that are incompatible with `KModel`'s module architecture.

### Training scripts

- `StyleTTS2/train_first.py` — Stage 1: decoder + alignment (reads `epochs_1st` from config)
- `StyleTTS2/train_second.py` — Stage 2: prosody predictor (reads `epochs_2nd` from config)
- `scripts/extract_voicepack.py` — Extract voicepack from trained checkpoint
- `scripts/test_inference.py` — Convert checkpoint to KModel format + run test sentences

### Key patches applied to upstream StyleTTS2

All patches are committed on the `main` branch of `semidark/StyleTTS2`:

- `text_utils.py` — imports from `kokoro_symbols.py` instead of default symbols
- `meldataset.py` — filters sequences > 510 tokens (PLBERT max_position_embeddings)
- `train_first.py` / `train_second.py` — checkpoint save moved before TensorBoard audio generation, F0 unsqueeze bug fixed, torch.load monkey-patch for PyTorch 2.6+, ipdb breakpoint removed
- `models.py` / `Modules/*.py` — weight_norm/spectral_norm migrated to new parametrizations API

### Critical Stage 2 Fixes (April 2026)

These fixes resolved the "static noise" and "style encoder collapse" issues that previously blocked Stage 2 training:

- **`train_second.py` (DataParallel order)** — Moved `MyDataParallel` wrapping to *after* `load_checkpoint()`. Wrapping before checkpoint loading prepends `module.` to all state dict keys, causing silent weight loading failures (`strict=False`) and random initialization. This was the root cause of the style encoder outputting ~1e17 values (static noise).

- **`train_second.py` (Weight loading)** — Removed `bert`, `bert_encoder`, and `predictor` from `ignore_modules`. Previously these modules were discarded during Stage 2 loading, forcing the prosody predictor to train from random initialization instead of fine-tuning the pretrained English Kokoro weights.

- **`train_second.py` (Adversarial logic)** — Restored accidentally deleted `y_rec_gt` / `y_rec_gt_pred` computation required for WavLM adversarial (`slmadv`) loss. Without this, enabling `joint_epoch` caused immediate crashes.

- **`train_second.py` (GAN gating)** — Changed discriminator activation (`start_ds`) to gate on `joint_epoch` instead of `diff_epoch`. Since Kokoro sets `diff_epoch=999` (disabling diffusion), the GAN discriminator never activated under the old logic.

- **`Modules/slmadv.py` (Diffusion bypass)** — Added `diffusion_enabled` flag to bypass the diffusion style sampler when diffusion is disabled (`diff_epoch=999`). Previously the sampler ran anyway, feeding garbage style embeddings to the discriminator.

- **TensorBoard audio (Kokoro-faithful inference)** — Replaced misleading ground-truth audio generation with proper Kokoro inference: predict duration/F0/energy using the `predictor`, extract mini voicepack from current `style_encoder`, and decode through the decoder. This gives accurate previews of actual inference quality rather than artificially-perfect reconstructions.

### Stage 2 Configuration Requirements

To prevent model collapse, Stage 2 **requires** these config settings:

```yaml
# In loss_params:
joint_epoch: 3              # Start adversarial training at epoch 3 (not 999)
lambda_slm: 1.0             # Enable SLM adversarial loss

# In training section:
second_stage_load_pretrained: false  # Load from first_stage.pth (trained)
                                      # true = load from kokoro_base.pth (not recommended)
```

**Why these matter:**
- `joint_epoch: 999` disables the GAN/SLM losses that provide gradient signal to the `style_encoder`. Without this signal, `spectral_norm` buffers drift and output explodes.
- `second_stage_load_pretrained: true` loads from the base English checkpoint, discarding your Stage 1 acoustic training. Must be `false` to continue from `first_stage.pth`.
