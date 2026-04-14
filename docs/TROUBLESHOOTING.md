# Troubleshooting

This document collects the critical training failures found during German Kokoro fine-tuning and how they were fixed.

If you are looking for the step-by-step path, use `TRAINING_GUIDE.md`.

## Stage 2 Static Noise / Collapsed Style Encoder

### Symptoms

- Stage 1 sounds usable, Stage 2 outputs static noise
- Voicepack norm collapses or explodes
- Stage 2 mel starts in a bad regime (acts like random init)

### Root Causes and Fixes

1) DataParallel wrap order was wrong in `train_second.py`
- Cause: wrapping model before checkpoint load changed key names (`module.` prefix)
- Fix: load checkpoint first, wrap after

2) `ignore_modules` excluded pretrained prosody modules
- Cause: `bert`, `bert_encoder`, `predictor` were dropped
- Fix: stop excluding those modules

3) Missing adversarial ground-truth tensors
- Cause: `y_rec_gt` / `y_rec_gt_pred` path had been removed
- Fix: restore computation before adversarial loss section

4) GAN discriminator gated on wrong epoch variable
- Cause: gated on `diff_epoch` instead of `joint_epoch`
- Fix: gate on `joint_epoch`

5) Diffusion sampler used even when diffusion was disabled
- Cause: invalid style embeddings fed to discriminator
- Fix: bypass diffusion sampling when diffusion is disabled

## Known Issues and Fixes

### 1) F0 shape mismatch at epoch boundaries

Files:
- `StyleTTS2/train_first.py`
- `StyleTTS2/train_second.py`

Fix:
- Remove the extra `unsqueeze(0)` on `F0_real` in TensorBoard audio generation path.

### 2) Checkpoint saved too late (after audio generation)

File:
- `StyleTTS2/train_first.py`

Fix:
- Save checkpoint before TensorBoard audio generation block.

### 3) Missing `.train()` restoration after checkpoint load

File:
- `StyleTTS2/train_second.py`

Fix:
- Restore train mode for all relevant modules, not only a subset.

### 4) Hardcoded debugger breakpoint

File:
- `StyleTTS2/train_second.py`

Fix:
- Remove `set_trace()` calls that hang under `accelerate launch`.

### 5) PLBERT max sequence overflow

File:
- `StyleTTS2/meldataset.py`

Fix:
- Filter samples whose cleaned token length exceeds 510.

### 6) PyTorch 2.6+ `torch.load` default changed

Files:
- `StyleTTS2/train_first.py`
- `StyleTTS2/train_second.py`

Fix:
- Ensure legacy checkpoints are loaded with `weights_only=False`.

## Validation Signals That Things Are Healthy

- Stage 2 starts from trained behavior, not random-init behavior
- Losses stay finite
- Voicepack norms remain stable and reasonable
- TensorBoard audio improves over epochs

## Recommended Debug Flow

1. Confirm symbol mapping compatibility first
2. Confirm checkpoint load paths and key matching
3. Confirm train/eval mode transitions
4. Confirm adversarial gating and diffusion bypass settings
5. Run a short smoke run before full training
