# kokoro-deutsch

Fine-tuning [Kokoro-82M](https://github.com/hexgrad/kokoro) for German text-to-speech using [StyleTTS2](https://github.com/yl4579/StyleTTS2).

## Status: Fully Functional Training Workflow

This project provides a **complete, validated training recipe** for fine-tuning Kokoro-82M on a new language. Both training stages have been debugged and are now fully functional, producing intelligible speech through the full Kokoro inference pipeline.

This is the **first fully functional, publicly accessible training workflow for Kokoro TTS**. These fixes provide a blueprint for anyone looking to train Kokoro for any language — not just German.

The end-to-end pipeline is confirmed working:

```
Dataset preparation -> Weight conversion -> Stage 1 training -> Stage 2 training -> Voicepack extraction -> KModel inference
```

### What works

- ✅ Dataset preparation pipeline (phonemization, audio conversion, train/val split)
- ✅ Weight conversion from Kokoro HuggingFace format to StyleTTS2 checkpoint format
- ✅ Stage 1 training (decoder + alignment) producing intelligible speech
- ✅ Stage 2 training (prosody predictor) with proper checkpoint loading and adversarial losses
- ✅ Voicepack extraction from trained checkpoints
- ✅ Kokoro-faithful TensorBoard audio monitoring during training
- ✅ Full inference through Kokoro's `KModel` / `KPipeline`

### Key fixes that made Stage 2 work

1. **DataParallel bug**: Fixed critical bug where `MyDataParallel` wrapping happened *before* checkpoint loading, causing silent weight loading failure (state dict keys didn't match with `module.` prefix)
2. **Missing pretrained weights**: Removed `bert`, `bert_encoder`, `predictor` from `ignore_modules` so Stage 2 actually fine-tunes the pretrained English Kokoro weights instead of starting from random initialization
3. **Missing adversarial logic**: Restored accidentally deleted `y_rec_gt` / `y_rec_gt_pred` computation that enables `joint_epoch` training
4. **GAN discriminator activation**: Fixed discriminator activation to gate on `joint_epoch` instead of `diff_epoch`
5. **Diffusion sampler bypass**: Added `diffusion_enabled` flag to use ground-truth style vectors when diffusion is disabled

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for the complete technical breakdown.

## Published Models

### 🎉 German Multi-Speaker Base Model (Stage 1)

**dida-80b/kokoro-deutsch-hui-base** is now available on HuggingFace!

A speaker-neutral German Stage 1 base model built on Kokoro-82M, trained on 51 speakers from the HUI Audio Corpus (CC0).

| Specification | Value |
|---|---|
| **Speakers** | 51 (24M / 27F) |
| **Training Audio** | ~51 hours (effective) |
| **Train Samples** | 20,495 |
| **Val Samples** | 418 |
| **Final Mel Loss** | 0.3264 (−44% from start) |
| **License** | CC0-1.0 |
| **Model** | [dida-80b/kokoro-deutsch-hui-base](https://huggingface.co/dida-80b/kokoro-deutsch-hui-base) |
| **Dataset** | [dida-80b/hui-german-51speakers](https://huggingface.co/datasets/dida-80b/hui-german-51speakers) |

**This is a base model, not a voice.** Use it as the foundation for Stage 2 fine-tuning with your own speaker data (2–5 hours of recordings). A base model trained on a single voice is biased toward that voice — this one isn't.

### Training Methodology

**Quality Filtering**
- Automated filter: duration 1–20s, RMS ≥−42 dB, clipping ≤0.1%, silence ≤50%
- Manual review: all 51 speakers individually verified
- Minimum threshold: 5 minutes per speaker (per Fraunhofer/IIS research on speaker embedding convergence)

**Duration Capping**
- 60 minutes per speaker to prevent dominant speakers from biasing the model
- Bernd_Ungerer alone contributed 81h raw — capped to 60h
- 60 min/speaker is 3.6× the VCTK gold standard (~24 min/speaker)

**Weighted Sampling**
- Small speakers duplicated so all 51 appear equally often in batches
- Formula: `weight = 60 min / speaker_duration`
- All speaker embeddings receive identical gradient update rates

See [issue #9](https://github.com/semidark/kokoro-deutsch/issues/9) for the complete training log and methodology.

## Dataset

The initial training used 11,551 clips (28.6 hours) of German speech. This dataset is **not included** in the repository and cannot be redistributed.

To reproduce, you can use any clean German speech dataset. Recommended open alternatives:

| Dataset | Hours | Speakers | License |
|---------|-------|----------|---------|
| [Thorsten-Voice](https://www.thorsten-voice.de/en/datasets-2/) | ~23h | 1 (male) | CC0 |
| [HUI-Audio-Corpus-German](https://opendata.iisys.de/dataset/hui-audio-corpus-german/) | ~200h | 5+ | Public Domain |
| [CSS10 German](https://github.com/Kyubyong/css10) | ~17h | 1 (female) | Public Domain |

See the [Training Guide - Dataset Preparation](docs/TRAINING_GUIDE.md#step-1-dataset-preparation) for format requirements.

## Quick Start

### Prerequisites

```bash
# System dependency (required for German G2P)
# Ubuntu/Debian:
sudo apt-get install espeak-ng

# macOS:
brew install espeak-ng
```

### Setup

```bash
git clone --recurse-submodules https://github.com/semidark/kokoro-deutsch
cd kokoro-deutsch

# Install Python dependencies
uv sync
```

### Training

The full training process is documented in **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**, covering:

1. Dataset preparation (audio format, IPA phonemization, train/val split)
2. Weight conversion (Kokoro HuggingFace -> StyleTTS2 format)
3. Symbol mapping (critical: Kokoro vs StyleTTS2 token indices)
4. `weight_norm` API migration (critical: old vs new PyTorch API)
5. Stage 1 training (decoder + alignment)
6. Stage 2 training (prosody predictor) and known pitfalls
7. Voicepack extraction
8. AMD ROCm-specific notes

### Fine-Tuning from the Base Model

To create your own German Kokoro voice using the published base model:

```bash
# Download the base checkpoint
wget https://huggingface.co/dida-80b/kokoro-deutsch-hui-base/resolve/main/first_stage.pth

# Place it in your training directory
mv first_stage.pth StyleTTS2/weights/

# Configure Stage 2 training
# In configs/config_german_ft.yml:
#   pretrained_model: first_stage.pth
#   second_stage_load_pretrained: false
#   joint_epoch: 3  # Enable adversarial training
```

Train Stage 2 with 2–5 hours of your own recordings to get a personalized German voice.

### Inference (once trained)

```bash
uv run kokoro --text "Hallo Welt" -o output.wav -l d --voice dm_daniel
```

## Repository Structure

```
kokoro/              # Kokoro TTS inference package (from hexgrad/kokoro, lightly modified)
kokoro.js/           # Kokoro JS/TS package (from hexgrad/kokoro, unmodified)
StyleTTS2/           # Training code (git submodule from semidark/StyleTTS2, main branch)
scripts/             # Original: dataset prep, voicepack extraction, inference testing
configs/             # Original: training configuration
docs/                # Original: training guide and documentation
training/            # Training data lists and config (audio files excluded)
demo/                # Gradio web demo (from hexgrad/kokoro, unmodified)
examples/            # Usage examples (from hexgrad/kokoro, unmodified)
tests/               # Unit tests (from hexgrad/kokoro, unmodified)
```

See [NOTICE](NOTICE) for full attribution of all upstream code.

## Contributing

This project exists because training a German voice for Kokoro turned out to be much harder than expected, and the training recipe didn't exist anywhere. The core pipeline is now complete and functional. Contributions are welcome — especially on:

- Training with open datasets (Thorsten-Voice, HUI-Audio-Corpus)
- Multi-speaker support
- Fine-tuning for other languages
- Improving audio quality through longer training runs

See the [GitHub Issues](https://github.com/semidark/kokoro-deutsch/issues) for the current discussion.

## Acknowledgements

- [hexgrad](https://github.com/hexgrad) for [Kokoro](https://github.com/hexgrad/kokoro) and [misaki](https://github.com/hexgrad/misaki)
- [yl4579](https://github.com/yl4579) for [StyleTTS 2](https://github.com/yl4579/StyleTTS2)
- [dida-80b](https://github.com/dida-80b) for training and publishing the [German multi-speaker base model](https://huggingface.co/dida-80b/kokoro-deutsch-hui-base)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
