# Training Guide

This guide is the practical path for fine-tuning Kokoro-82M for German.

For deep debugging details, see `TROUBLESHOOTING.md`.
For architecture and compatibility notes, see `ARCHITECTURE.md`.

## 1) Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng libsndfile1

# macOS
brew install espeak-ng libsndfile
```

- Python: 3.10-3.13
- Package manager: `uv`
- Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/semidark/kokoro-deutsch
cd kokoro-deutsch
uv sync
```

## 2) Prepare Dataset

Create file lists in StyleTTS2 format:

`path/to/audio.wav|IPA phoneme string|speaker_name`

Requirements:
- WAV, mono, 24kHz, 16-bit
- Typical clip duration: 2-30s
- Keep phoneme strings compatible with Kokoro symbols

German G2P example:

```python
from misaki import espeak

g2p = espeak.EspeakG2P(language='de')
phonemes, _ = g2p(text)
phonemes = phonemes.replace('ʏ', 'y')
```

Use:
- `scripts/prepare_dataset.py`
- `scripts/prepare_training.py`

## 3) Prepare Base Weights

Convert Kokoro HuggingFace weights into StyleTTS2-compatible checkpoint format.

Expected output:
- `training/kokoro_base.pth`

Keep `load_only_params: true` for initial loading.

## 4) Symbol Mapping (Critical)

StyleTTS2 default token indices do not match Kokoro token indices.

Required:
- `StyleTTS2/text_utils.py` must import from `kokoro_symbols.py`
- `kokoro_symbols.py` must contain the 178-token Kokoro mapping

Without this, training appears to run but token embeddings are silently wrong.

## 5) StyleTTS2 Environment

`StyleTTS2/` is a patched submodule with required fixes already included.

You still need utility models:
- `Utils/JDC/bst.t7`
- `Utils/ASR/config.yml` and `Utils/ASR/epoch_00080.pth`
- `Utils/PLBERT/*`

Build monotonic alignment extension:

```bash
cd StyleTTS2/monotonic_align
python setup.py build_ext --inplace
```

## 6) Configure Training

Primary config: `configs/config_german_ft.yml`

Top-level keys used by training scripts include:
- `batch_size`
- `epochs_1st`
- `epochs_2nd`
- `save_freq`
- `pretrained_model`
- `load_only_params`
- `first_stage_path`

Important Stage 2 settings:
- `second_stage_load_pretrained: false`
- `joint_epoch: 3`
- `lambda_slm: 1.0`

## 7) Smoke Test

Before long runs:
- Verify symbol map loads and has length 178
- Verify model loads without size mismatches
- Run a short training start and confirm finite losses

## 8) Stage 1 Training

Run from `StyleTTS2/`:

```bash
accelerate launch train_first.py --config_path ../configs/config_german_ft.yml
```

Expected trend:
- Mel loss should drop over epochs
- Checkpoints saved in `StyleTTS2/logs/<run>/`

## 9) Stage 2 Training

Run from `StyleTTS2/`:

```bash
accelerate launch train_second.py --config_path ../configs/config_german_ft.yml
```

Healthy sign of proper loading:
- Stage 2 mel starts low (around the trained regime, not random-init regime)

## 10) Extract Voicepack and Test Inference

Extract:

```bash
python scripts/extract_voicepack.py \
  --model StyleTTS2/logs/kokoro-deutsch/epoch_2nd_00009.pth \
  --audio-dir path/to/audio \
  --output voices/dm_daniel.pt
```

Convert/test inference:

```bash
python scripts/test_inference.py \
  --checkpoint StyleTTS2/logs/kokoro-deutsch/epoch_2nd_00009.pth \
  --voicepack voices/dm_daniel.pt \
  --output-dir test_output/
```

## 11) Quick Checklist

- [ ] Dataset lists are valid
- [ ] Symbol mapping is Kokoro-compatible
- [ ] Utility models downloaded
- [ ] Stage 1 completes and checkpoints save
- [ ] Stage 2 starts from trained weights
- [ ] Voicepack extraction succeeds
- [ ] Inference audio is intelligible
