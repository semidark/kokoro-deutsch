#!/usr/bin/env python3
"""
Kokoro German: Extract Voicepack
=================================
Extracts a voicepack (.pt) from a fine-tuned StyleTTS2 checkpoint by running
both style encoders (acoustic + prosodic) on representative utterances and
averaging the resulting style vectors.

Usage:
    # Single checkpoint (Stage 1 only — uses style_encoder for both halves)
    python scripts/extract_voicepack.py \
        --model StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \
        --audio-dir dataset/audio/dm_daniel \
        --output voices/dm_daniel.pt

    # Two checkpoints (recommended after Stage 2 training):
    #   style_encoder from Stage 1, predictor_encoder from Stage 2
    python scripts/extract_voicepack.py \
        --model StyleTTS2/logs/kokoro_german/epoch_2nd_00001.pth \
        --style-encoder-model StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \
        --audio-dir dataset/audio/dm_daniel \
        --output voices/dm_daniel.pt

    # CPU (slower but works without GPU / while GPU is busy training)
    python scripts/extract_voicepack.py \
        --model StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \
        --audio-dir dataset/audio/dm_daniel \
        --output voices/dm_daniel.pt \
        --device cpu

Voicepack format: tensor of shape [510, 1, 256] (float32)
  - 510 = max phoneme sequence length
  - 256 = style_dim * 2 (128 for decoder/timbre + 128 for predictor/prosody)
  - First 128 dims come from style_encoder (acoustic/timbre)
  - Last 128 dims come from predictor_encoder (prosody)
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


# ── StyleTTS2 model components (standalone, no external imports) ─────────


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    groups=dim_in,
                    padding=(1, 0),
                )
            )
        elif self.layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    groups=dim_in,
                    padding=1,
                )
            )
        else:
            raise RuntimeError(f"Unexpected downsample type: {self.layer_type}")

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(f"Unexpected downsample type: {self.layer_type}")


class ResBlk(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        downsample="none",
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(
                nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
            )

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
        return s


# ── Main extraction logic ───────────────────────────────────────────────


def extract_voicepack(
    model_path: str,
    audio_dir: str,
    output_path: str,
    num_samples: int = 200,
    device: str = "auto",
    style_encoder_model: str = None,
):
    import random

    import soundfile as sf
    import torchaudio

    # ── Resolve device ───────────────────────────────────────────────────
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # ── Collect audio files ──────────────────────────────────────────────
    audio_path = Path(audio_dir)
    wav_files = sorted(audio_path.glob("*.wav"))
    if not wav_files:
        print(f"ERROR: No WAV files found in {audio_dir}")
        sys.exit(1)

    print(f"Found {len(wav_files):,} WAV files in {audio_dir}")

    # Sample a subset for embedding
    rng = random.Random(42)
    if len(wav_files) > num_samples:
        wav_files = rng.sample(wav_files, num_samples)
    print(f"Using {len(wav_files)} samples for voicepack extraction")

    # ── Build style encoders and load weights ────────────────────────────

    def strip_prefix(state_dict, prefix="module."):
        """Strip DataParallel 'module.' prefix from state dict keys if present."""
        if any(k.startswith(prefix) for k in state_dict.keys()):
            return {
                k[len(prefix) :] if k.startswith(prefix) else k: v
                for k, v in state_dict.items()
            }
        return state_dict

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    net = checkpoint["net"]

    # Kokoro-82M architecture: dim_in=64, style_dim=128, hidden_dim=512
    style_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)
    predictor_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)

    # Load style_encoder weights — from a separate checkpoint if provided,
    # otherwise from the main checkpoint. Stage 2 training can degrade the
    # style_encoder (spectral_norm buffer drift), so loading from Stage 1
    # is recommended when using a Stage 2 checkpoint.
    if style_encoder_model:
        print(f"Loading style_encoder from separate checkpoint: {style_encoder_model}")
        se_checkpoint = torch.load(
            style_encoder_model, map_location="cpu", weights_only=False
        )
        se_net = se_checkpoint["net"]
        style_encoder.load_state_dict(strip_prefix(se_net["style_encoder"]))
        se_epoch = se_checkpoint.get("epoch", "?")
        print(f"  style_encoder loaded (epoch {se_epoch})")
    else:
        style_encoder.load_state_dict(strip_prefix(net["style_encoder"]))

    # predictor_encoder is only trained in Stage 2 (train_second.py).
    # After Stage 1 only, its weights are randomly initialized and produce
    # exploded outputs even though parameter norms look reasonable.
    # Detect this by checking output norm on a dummy input. If exploded,
    # fall back to using style_encoder for both halves of the voicepack
    # (same initialization Stage 2 uses — it deep-copies style_encoder).
    predictor_encoder_trained = True
    try:
        predictor_encoder.load_state_dict(strip_prefix(net["predictor_encoder"]))
        # Test output norm on a dummy mel-like input
        with torch.no_grad():
            dummy = torch.randn(1, 1, 80, 200)
            test_out = predictor_encoder(dummy)
            if test_out.norm().item() > 1e3:
                predictor_encoder_trained = False
    except Exception:
        predictor_encoder_trained = False

    if not predictor_encoder_trained:
        print("  predictor_encoder appears untrained (Stage 1 only)")
        print("  Using style_encoder for both acoustic and prosodic embeddings")
        predictor_encoder.load_state_dict(
            strip_prefix(se_net["style_encoder"])
            if style_encoder_model
            else strip_prefix(net["style_encoder"])
        )

    style_encoder = style_encoder.to(device).eval()
    predictor_encoder = predictor_encoder.to(device).eval()

    # Validate style encoder output norms
    with torch.no_grad():
        dummy = torch.randn(1, 1, 80, 200).to(device)
        se_norm = style_encoder(dummy).norm().item()
        pe_norm = predictor_encoder(dummy).norm().item()
        print(f"  style_encoder output norm (dummy): {se_norm:.4f}")
        print(f"  predictor_encoder output norm (dummy): {pe_norm:.4f}")
        if se_norm < 0.5:
            print(f"  WARNING: style_encoder norm is very low ({se_norm:.4f}).")
            print(f"  This may indicate a collapsed encoder. Consider using")
            print(f"  --style-encoder-model to load from a Stage 1 checkpoint.")

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", 0)
    print(f"Loaded encoders from checkpoint")
    print(f"  Epoch: {epoch}, Val loss: {val_loss:.4f}")

    # ── Mel spectrogram transform ────────────────────────────────────────
    # Must match StyleTTS2's training preprocessing exactly:
    #   mel = to_mel(waveform)   # n_fft=2048, win=1200, hop=300, n_mels=80
    #   mel = (log(1e-5 + mel) - mean) / std   # mean=-4, std=4
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
    ).to(device)

    mel_mean = -4
    mel_std = 4

    # ── Extract style vectors ────────────────────────────────────────────
    acoustic_styles = []
    prosodic_styles = []
    skipped = 0

    print(f"\nExtracting style vectors...")
    with torch.no_grad():
        for i, wav_path in enumerate(wav_files):
            # Load audio via soundfile (avoids torchcodec dependency)
            data, sr = sf.read(str(wav_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)  # mono
            if sr != 24000:
                # Resample using torchaudio
                waveform = torch.from_numpy(data).unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, sr, 24000)
            else:
                waveform = torch.from_numpy(data).unsqueeze(0)  # [1, samples]

            waveform = waveform.to(device)

            # Compute mel spectrogram (same as StyleTTS2's preprocess())
            mel = mel_transform(waveform)  # [1, 80, T]
            mel = (torch.log(1e-5 + mel) - mel_mean) / mel_std  # normalize

            # Skip clips with very short mels (style encoder needs >= 80 frames)
            if mel.shape[-1] < 80:
                skipped += 1
                continue

            mel_input = mel.unsqueeze(1)  # [1, 1, 80, T] — add channel dim

            # Run both style encoders
            s_acoustic = style_encoder(mel_input)  # [1, 128]
            s_prosodic = predictor_encoder(mel_input)  # [1, 128]

            acoustic_styles.append(s_acoustic.cpu())
            prosodic_styles.append(s_prosodic.cpu())

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(wav_files)} files...")

    if skipped > 0:
        print(f"  Skipped {skipped} files (too short)")

    print(f"  Extracted {len(acoustic_styles)} style vectors")

    if not acoustic_styles:
        print("ERROR: No style vectors extracted. Check audio files.")
        sys.exit(1)

    # ── Average and build voicepack ──────────────────────────────────────
    avg_acoustic = torch.cat(acoustic_styles, dim=0).mean(dim=0)  # [128]
    avg_prosodic = torch.cat(prosodic_styles, dim=0).mean(dim=0)  # [128]

    # Kokoro voicepack format: [510, 1, 256]
    # First 128 dims = acoustic (decoder conditioning / timbre)
    # Last 128 dims = prosodic (predictor conditioning / prosody)
    combined = torch.cat([avg_acoustic, avg_prosodic], dim=0)  # [256]
    voicepack = combined.unsqueeze(0).unsqueeze(0).expand(510, 1, 256).clone()

    print(f"\nVoicepack shape: {tuple(voicepack.shape)}")
    print(f"  Acoustic style norm: {avg_acoustic.norm():.4f}")
    print(f"  Prosodic style norm: {avg_prosodic.norm():.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(voicepack, str(output))
    print(f"\nSaved voicepack: {output} ({output.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract voicepack from fine-tuned Kokoro model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1 only (uses style_encoder for both halves)
  python scripts/extract_voicepack.py \\
      --model StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \\
      --audio-dir dataset/audio/dm_daniel \\
      --output voices/dm_daniel.pt

  # After Stage 2 (recommended: style_encoder from Stage 1, predictor_encoder from Stage 2)
  python scripts/extract_voicepack.py \\
      --model StyleTTS2/logs/kokoro_german/epoch_2nd_00001.pth \\
      --style-encoder-model StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \\
      --audio-dir dataset/audio/dm_daniel \\
      --output voices/dm_daniel.pt

  # CPU (while GPU is busy training)
  python scripts/extract_voicepack.py \\
      --model StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \\
      --device cpu
""",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to fine-tuned StyleTTS2 checkpoint (.pth). Used for predictor_encoder "
        "(and style_encoder unless --style-encoder-model is specified).",
    )
    parser.add_argument(
        "--style-encoder-model",
        default=None,
        help="Path to a separate checkpoint for loading the style_encoder weights. "
        "Recommended: use a Stage 1 checkpoint here when --model is a Stage 2 "
        "checkpoint, because Stage 2 can degrade the style_encoder.",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing speaker's WAV files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output voicepack path (.pt)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of audio samples to use for extraction (default: 200)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on: auto (default), cpu, or cuda",
    )

    args = parser.parse_args()
    extract_voicepack(
        model_path=args.model,
        audio_dir=args.audio_dir,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device,
        style_encoder_model=args.style_encoder_model,
    )


if __name__ == "__main__":
    main()
