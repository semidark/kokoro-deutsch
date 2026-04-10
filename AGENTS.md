# AGENTS.md — kokoro-deutsch

## Project Overview

kokoro-deutsch is a training recipe for fine-tuning [Kokoro TTS](https://github.com/hexgrad/kokoro) (82M parameters, based on StyleTTS 2) for German. The project contains:

- A lightly modified copy of the `kokoro/` inference package (German language code added)
- A patched fork of `StyleTTS2/` as a git submodule (`semidark/StyleTTS2`, branch `main`)
- Original scripts for dataset preparation, voicepack extraction, and inference testing
- A comprehensive training guide documenting every step and bug fix

Unmodified upstream code (`kokoro.js/`, `demo/`, `examples/`, `tests/`) is included for convenience. See `NOTICE` for full attribution.

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

```bash
# Run all Python tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_custom_stft.py

# Run a single test function
uv run pytest tests/test_custom_stft.py::test_stft_reconstruction

# Run with verbose output
uv run pytest tests/test_custom_stft.py -v
```

There is no pytest configuration in `pyproject.toml` — default pytest settings apply.

### JavaScript tests (kokoro.js/)

```bash
# From kokoro.js/ directory
npm test          # runs: vitest run
npm run build     # rollup + tsc
npm run format    # prettier --write . --print-width 1000
```

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

### JavaScript (kokoro.js/)

- **Formatter:** Prettier with `--print-width 1000` (very long lines are intentional).
- **Test framework:** vitest.
- **Module system:** ES modules (`"type": "module"`).
- **Build:** Rollup (CJS + ESM + web bundle).

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
docs/                # Training guide and documentation
  TRAINING_GUIDE.md  # Comprehensive fine-tuning guide
  images/            # TensorBoard screenshots
tests/               # Python tests (from hexgrad/kokoro, unmodified)
demo/                # Gradio web demo (from hexgrad/kokoro, unmodified)
examples/            # Usage examples (from hexgrad/kokoro, unmodified)
kokoro.js/           # JS/TS package (from hexgrad/kokoro, unmodified)
voices/              # Voicepack .pt files (excluded from git)
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
