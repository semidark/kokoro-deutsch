# Architecture and Compatibility Notes

Technical reference for Kokoro-82M fine-tuning compatibility.

For how-to training steps, use `TRAINING_GUIDE.md`.

## Kokoro-82M Component Layout

Reference component sizes used for checkpoint compatibility checks:

| Component | Parameters |
|---|---|
| bert (PLBERT) | 6.29M |
| bert_encoder | 0.39M |
| predictor | 16.19M |
| text_encoder | 5.61M |
| decoder (ISTFTNet) | 53.28M |
| Total | 81.76M |

Voicepack target shape:
- `[510, 1, 256]` (float32)

## weight_norm API Compatibility

### Why it matters

Old API (`torch.nn.utils.weight_norm`) and new API (`torch.nn.utils.parametrizations.weight_norm`) create different state-dict key layouts.

If StyleTTS2 is trained with old API and inference expects new API, checkpoint loading can be brittle and may fail silently under non-strict loading paths.

### Required status

StyleTTS2 patched files must use new parametrizations API:
- `StyleTTS2/models.py`
- `StyleTTS2/Modules/istftnet.py`
- `StyleTTS2/Modules/hifigan.py`
- `StyleTTS2/Modules/discriminators.py`

## Symbol Mapping Compatibility

Kokoro and default StyleTTS2 use different token index assignments.

Implication:
- same symbol set size does not imply index compatibility

Requirement:
- `StyleTTS2/text_utils.py` must use Kokoro mapping (`kokoro_symbols.py`)

## German G2P Notes

- G2P backend: `misaki` + `espeak-ng`
- German code path uses `espeak.EspeakG2P(language='de')`
- Symbol `ʏ` is not in Kokoro vocab and must be normalized to `y`

## Sequence Length Constraint

- PLBERT max position embeddings: 512
- Practical training cap: 510 cleaned tokens

Samples above this should be filtered before batching.

## Runtime Notes (AMD ROCm)

- fp32 recommended for stability
- mixed precision can be unstable depending on stack/hardware
- first run may spend significant time compiling kernels

## Inference Packaging Notes

When exporting trained checkpoints for `KModel`, ensure the expected components are present and keys align with Kokoro inference code:
- `bert`
- `bert_encoder`
- `predictor`
- `text_encoder`
- `decoder`

Use `scripts/test_inference.py` to verify conversion and produce sample outputs.
