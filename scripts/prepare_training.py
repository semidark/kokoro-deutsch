#!/usr/bin/env python3
"""
Kokoro German: Prepare Training Data
=====================================
Converts the dataset produced by prepare_dataset.py into the format
expected by StyleTTS2's training scripts.

Run this on any CUDA/ROCm machine before starting training.

Usage:
    # Step 1: Generate train/val lists and precompute features
    python scripts/prepare_training.py prepare

    # Step 2: Convert Kokoro weights to StyleTTS2 format
    python scripts/prepare_training.py convert-weights

    # Step 3: Smoke test (verify data loads correctly)
    python scripts/prepare_training.py verify
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("./dataset")
TRAINING_DIR = Path("./training")
WAVS_DIR = DATASET_DIR / "audio"
METADATA_FILE = DATASET_DIR / "metadata.csv"
PHONEMES_FILE = DATASET_DIR / "phonemes.csv"

TRAIN_LIST = TRAINING_DIR / "train_list.txt"
VAL_LIST = TRAINING_DIR / "val_list.txt"
OOD_FILE = TRAINING_DIR / "OOD_texts.txt"
MELS_DIR = TRAINING_DIR / "mels"
F0_DIR = TRAINING_DIR / "f0"

# ── Audio params (must match Kokoro/StyleTTS2 config) ────────────────────────

SAMPLE_RATE = 24000
N_FFT = 2048
HOP_LENGTH = 300
WIN_LENGTH = 1200
N_MELS = 80
F_MIN = 0
F_MAX = 8000

# ── Split ────────────────────────────────────────────────────────────────────

VAL_RATIO = 0.05
RANDOM_SEED = 42


def cmd_prepare():
    """Generate train/val lists and optionally precompute mels and F0."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load metadata and phonemes ───────────────────────────────────────
    if not METADATA_FILE.exists() or not PHONEMES_FILE.exists():
        print("ERROR: metadata.csv or phonemes.csv not found.")
        print(
            "Run: uv run python scripts/prepare_dataset.py format --rename-speakers d_speaker1=dm_daniel"
        )
        sys.exit(1)

    # Parse metadata: filename|text|speaker
    meta = {}
    with open(METADATA_FILE) as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                filename, text, speaker = parts
                meta[filename] = {"text": text, "speaker": speaker}

    # Parse phonemes: filename|ipa
    phonemes = {}
    with open(PHONEMES_FILE) as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                filename, ipa = parts
                phonemes[filename] = ipa

    # Merge and validate
    entries = []
    missing_phonemes = 0
    missing_wav = 0
    empty_phonemes = 0
    for filename, m in meta.items():
        wav_path = WAVS_DIR / filename
        if not wav_path.exists():
            missing_wav += 1
            continue
        ipa = phonemes.get(filename, "")
        if not ipa:
            missing_phonemes += 1
            continue
        # Skip entries with very short phoneme sequences (likely bad transcriptions)
        if len(ipa) < 5:
            empty_phonemes += 1
            continue
        entries.append(
            {
                "filename": filename,
                "wav_path": str(wav_path),
                "text": m["text"],
                "speaker": m["speaker"],
                "ipa": ipa,
            }
        )

    print(f"Total entries: {len(meta):,}")
    print(f"Valid entries: {len(entries):,}")
    if missing_wav:
        print(f"  Missing WAV: {missing_wav}")
    if missing_phonemes:
        print(f"  Missing phonemes: {missing_phonemes}")
    if empty_phonemes:
        print(f"  Empty/short phonemes: {empty_phonemes}")

    if not entries:
        print("ERROR: No valid entries found.")
        sys.exit(1)

    # ── Train/val split ──────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(entries)
    n_val = max(1, int(len(entries) * VAL_RATIO))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    print(f"\nSplit: {len(train_entries):,} train / {len(val_entries):,} val")

    # ── Write train/val lists ────────────────────────────────────────────
    # Format: relative_wav_path|phoneme_sequence|speaker_id
    # Note: We use the wav path relative to the audio dir
    def write_list(path, entries_list):
        with open(path, "w") as f:
            for e in entries_list:
                # StyleTTS2 expects: filename|transcript|speaker
                # We provide IPA phonemes as the transcript
                f.write(f"{e['filename']}|{e['ipa']}|{e['speaker']}\n")

    write_list(TRAIN_LIST, train_entries)
    write_list(VAL_LIST, val_entries)

    print(f"Wrote {TRAIN_LIST} ({len(train_entries):,} lines)")
    print(f"Wrote {VAL_LIST} ({len(val_entries):,} lines)")

    # ── Write OOD texts (German sentences not in training) ───────────────
    ood_sentences = [
        "Die Bundesrepublik Deutschland ist ein demokratischer Staat.",
        "Morgen wird es regnen, nehmen Sie einen Regenschirm mit.",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Können Sie mir bitte den Weg zum Bahnhof zeigen?",
        "Die Kinder spielen fröhlich im Garten und lachen laut.",
        "Wissenschaftler haben eine bahnbrechende Entdeckung gemacht.",
        "Das Frühstück war ausgezeichnet, besonders die frischen Brötchen.",
        "Im Schwarzwald gibt es viele schöne Wanderwege zu entdecken.",
        "Die Universität bietet verschiedene Studiengänge für internationale Studenten an.",
        "Bitte vergessen Sie nicht, die Tür abzuschließen, wenn Sie gehen.",
        "Der Weihnachtsmarkt in Nürnberg ist weltberühmt für seinen Glühwein.",
        "Diese Aufgabe erfordert besondere Sorgfalt und Aufmerksamkeit.",
        "Die Zugverbindung zwischen München und Berlin dauert etwa vier Stunden.",
        "Könnten Sie mir erklären, wie dieses Gerät funktioniert?",
        "Das Unternehmen hat im vergangenen Quartal einen Rekordgewinn erzielt.",
        "Die Bibliothek hat montags bis freitags von acht bis zwanzig Uhr geöffnet.",
        "Entschuldigen Sie die Verspätung, der Verkehr war heute besonders schlimm.",
        "Die neue Brücke über den Rhein wird nächstes Jahr fertiggestellt.",
        "Haben Sie schon einmal die Berliner Philharmoniker live gehört?",
        "Der Arzt empfiehlt, täglich mindestens dreißig Minuten spazieren zu gehen.",
    ]

    # Convert OOD texts to IPA phonemes
    try:
        from misaki import espeak

        g2p = espeak.EspeakG2P(language="de")
        ood_phonemes = []
        for text in ood_sentences:
            try:
                ph, _ = g2p(text)
                ph = ph.replace("\u028f", "y")  # ʏ → y fixup
                ood_phonemes.append(ph)
            except Exception:
                pass

        with open(OOD_FILE, "w") as f:
            f.write("\n".join(ood_phonemes) + "\n")
        print(f"Wrote {OOD_FILE} ({len(ood_phonemes)} sentences)")
    except ImportError:
        print("WARNING: misaki not available, skipping OOD text generation.")
        print("  Generate OOD_texts.txt on a machine with misaki installed.")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Training data ready in {TRAINING_DIR}/")
    print(f"  train_list.txt  : {len(train_entries):,} entries")
    print(f"  val_list.txt    : {len(val_entries):,} entries")
    print(f"  OOD_texts.txt   : out-of-domain German sentences")
    print(f"  Audio dir       : {WAVS_DIR}/")
    print(f"{'=' * 60}")


def cmd_precompute():
    """Pre-compute mel spectrograms and F0 for all training audio.

    This is optional but saves significant time during GPU training.
    Run before starting training to save time during GPU training.
    """
    import numpy as np

    try:
        import torch
        import torchaudio
    except ImportError:
        print("ERROR: torch and torchaudio are required for pre-computing features.")
        print("Install: pip install torch torchaudio")
        sys.exit(1)

    from tqdm import tqdm

    MELS_DIR.mkdir(parents=True, exist_ok=True)
    F0_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all wav files from train + val lists
    wav_files = set()
    for list_file in [TRAIN_LIST, VAL_LIST]:
        if not list_file.exists():
            print(f"ERROR: {list_file} not found. Run 'prepare' first.")
            sys.exit(1)
        with open(list_file) as f:
            for line in f:
                filename = line.strip().split("|")[0]
                wav_files.add(filename)

    print(f"Pre-computing features for {len(wav_files):,} files...")

    # Set up mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=1.0,  # amplitude spectrogram
        normalized=False,
    )

    errors = 0
    skipped = 0
    computed = 0

    for filename in tqdm(sorted(wav_files), desc="Features"):
        wav_path = WAVS_DIR / filename
        mel_path = MELS_DIR / (filename.replace("/", "_").replace(".wav", ".npy"))
        f0_path = F0_DIR / (filename.replace("/", "_").replace(".wav", ".npy"))

        if mel_path.exists() and f0_path.exists():
            skipped += 1
            continue

        try:
            # Load audio
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Compute mel spectrogram
            mel = mel_transform(waveform)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            np.save(str(mel_path), mel.squeeze(0).numpy())

            # Compute F0 using pyin
            import librosa

            y = waveform.squeeze().numpy()
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            np.save(str(f0_path), f0)

            computed += 1
        except Exception as e:
            print(f"ERROR {filename}: {e}")
            errors += 1

    print(f"\nComputed: {computed}  Skipped: {skipped}  Errors: {errors}")
    print(f"Mels: {MELS_DIR}/")
    print(f"F0:   {F0_DIR}/")


def cmd_convert_weights(force: bool = False):
    """Convert Kokoro-82M weights to a format compatible with StyleTTS2 training.

    StyleTTS2's load_checkpoint expects:
        {'net': {component_name: state_dict, ...}, ...}

    Kokoro's weights are:
        {'bert': {'module.key': tensor}, 'predictor': {...}, ...}

    We strip the 'module.' prefix and wrap in {'net': ...}.
    """
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required. Install: pip install torch")
        sys.exit(1)

    from huggingface_hub import hf_hub_download

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_DIR / "kokoro_base.pth"

    if output_path.exists() and not force:
        print(f"Converted weights already exist: {output_path}")
        print("Use --force to regenerate.")
        return

    print("Downloading Kokoro-82M weights from HuggingFace...")
    model_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    config_path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")

    print(f"Loading weights from {model_path}...")
    kokoro_state = torch.load(model_path, map_location="cpu", weights_only=True)

    # Kokoro stores weights as:
    #   { 'bert': {'module.key': tensor}, 'predictor': {...}, ... }
    # StyleTTS2 expects:
    #   { 'net': {'bert': state_dict, 'predictor': state_dict, ...} }
    #
    # We strip the 'module.' prefix from all keys and wrap in 'net'.

    net = {}
    total_params = 0
    for component, state_dict in kokoro_state.items():
        cleaned = {}
        for key, tensor in state_dict.items():
            # Strip 'module.' prefix if present
            clean_key = key.removeprefix("module.")
            cleaned[clean_key] = tensor
            total_params += tensor.numel()
        net[component] = cleaned
        print(f"  {component}: {len(cleaned)} tensors")

    # Wrap in StyleTTS2's expected checkpoint format
    checkpoint = {
        "net": net,
    }

    # Also save config alongside
    import shutil

    config_out = TRAINING_DIR / "config.json"
    shutil.copy2(config_path, config_out)

    torch.save(checkpoint, output_path)
    print(f"\nSaved converted weights: {output_path}")
    print(
        f"  Format: {{'net': {{bert, bert_encoder, predictor, decoder, text_encoder}}}}"
    )
    print(f"Saved config: {config_out}")
    print(f"Total parameters: {total_params / 1e6:.2f}M")


def cmd_patch_styletts2():
    """Patch StyleTTS2's text_utils.py and meldataset.py to use Kokoro's vocab.

    This is CRITICAL: Kokoro and StyleTTS2 use the same 178 tokens but with
    different index assignments. If we don't fix this, the pre-trained
    embeddings will be scrambled (e.g., the model would think the German
    affricate ʦ is the letter D).

    This command generates a drop-in replacement for StyleTTS2's symbol
    mapping that matches Kokoro-82M's config.json exactly.
    """
    config_path = TRAINING_DIR / "config.json"
    if not config_path.exists():
        print("ERROR: training/config.json not found. Run 'convert-weights' first.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    vocab = config["vocab"]  # {symbol_str: int_id}
    n_token = config["n_token"]  # 178

    # Build the symbols list: index 0 = pad ($), rest from vocab sorted by ID
    # Kokoro's vocab has gaps (not all 0-177 are assigned), so we fill gaps
    # with placeholder characters that won't appear in real data.
    symbols = ["$"] * n_token  # index 0 is pad
    for symbol, idx in vocab.items():
        if 0 < idx < n_token:
            symbols[idx] = symbol

    # Fill any remaining gaps with unique placeholder chars
    # (these are unused embedding slots — they won't affect training)
    placeholder_base = 0xE000  # Unicode Private Use Area
    placeholder_idx = 0
    for i in range(n_token):
        if symbols[i] == "$" and i != 0:
            symbols[i] = chr(placeholder_base + placeholder_idx)
            placeholder_idx += 1

    # Verify: every Kokoro vocab entry should map to its correct index
    for symbol, expected_idx in vocab.items():
        actual_idx = symbols.index(symbol) if symbol in symbols else -1
        assert actual_idx == expected_idx, (
            f"Symbol {repr(symbol)} expected at index {expected_idx} but found at {actual_idx}"
        )

    print(f"Generated symbols list: {n_token} entries")
    print(f"  Mapped from vocab: {len(vocab)} symbols")
    print(f"  Gap placeholders:  {placeholder_idx}")

    # Generate the Python code for text_utils.py
    # We build a single symbols list string that can be directly used
    symbols_code = _generate_symbols_code(symbols, vocab)

    # Write the patch file
    patch_path = TRAINING_DIR / "kokoro_symbols.py"
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(symbols_code)

    print(f"\nWrote {patch_path}")
    print()
    print("To apply this patch to StyleTTS2, do ONE of the following:")
    print()
    print("  Option A: Copy the file into StyleTTS2/")
    print(f"    cp {patch_path} StyleTTS2/kokoro_symbols.py")
    print("    Then in StyleTTS2/text_utils.py, replace the symbols definition with:")
    print("      from kokoro_symbols import symbols")
    print("    And in StyleTTS2/meldataset.py, do the same.")
    print()
    print("  Option B: Manually replace the symbols list in text_utils.py")
    print(f"    See {patch_path} for the exact code to paste.")


def _generate_symbols_code(symbols, vocab):
    """Generate Python source code defining the symbols list matching Kokoro's vocab."""
    lines = [
        '"""',
        "Kokoro-82M Symbol Mapping for StyleTTS2",
        "=========================================",
        "Auto-generated from Kokoro-82M config.json.",
        "Replaces StyleTTS2's default symbol list in text_utils.py and meldataset.py.",
        "",
        "CRITICAL: Kokoro and StyleTTS2 use different index assignments for the same",
        "178-token vocabulary. This file provides the exact mapping that matches",
        "Kokoro-82M's pre-trained embeddings.",
        "",
        "Usage in StyleTTS2:",
        "  from kokoro_symbols import symbols, dicts",
        '"""',
        "",
        "# fmt: off",
        "# Kokoro-82M vocabulary: 178 tokens (index 0 = pad, gaps filled with PUA chars)",
        "symbols = [",
    ]

    # Write symbols with comments showing the index and hex code
    for i, sym in enumerate(symbols):
        if sym in vocab.values() if isinstance(sym, int) else sym in vocab:
            # This is a real symbol from Kokoro's vocab
            if sym == '"':
                repr_sym = "'\\\"'"
            elif sym == "'":
                repr_sym = '"\'"'
            elif sym == "\\":
                repr_sym = "'\\\\'"
            else:
                repr_sym = repr(sym)

            # Find if it's printable and safe to show
            if ord(sym) >= 32 and ord(sym) < 127:
                comment = f"  # {i:3d}: {sym}"
            else:
                comment = f"  # {i:3d}: U+{ord(sym):04X} ({sym})"
        else:
            repr_sym = repr(sym)
            if i == 0:
                comment = f"  # {i:3d}: PAD"
            else:
                comment = f"  # {i:3d}: (unused placeholder)"

        lines.append(f"    {repr_sym},{comment}")

    lines.extend(
        [
            "]",
            "# fmt: on",
            "",
            "# Build symbol-to-ID lookup dict (same interface as StyleTTS2's TextCleaner)",
            "dicts = {sym: i for i, sym in enumerate(symbols)}",
            "",
            "",
            "class TextCleaner:",
            '    """Drop-in replacement for StyleTTS2\'s TextCleaner using Kokoro vocab."""',
            "    def __init__(self, dummy=0):",
            "        self.word_index_dictionary = dicts",
            "",
            "    def __call__(self, text):",
            "        # Map each character to its index, skip unknown chars",
            "        indexes = []",
            "        for char in text:",
            "            if char in self.word_index_dictionary:",
            "                indexes.append(self.word_index_dictionary[char])",
            "        return indexes",
            "",
            "",
            f'assert len(symbols) == 178, f"Expected 178 symbols, got {{len(symbols)}}"',
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def cmd_verify():
    """Verify training data integrity — check a few samples end-to-end."""
    print("Verifying training data...")

    issues = []

    # Check files exist
    for f in [TRAIN_LIST, VAL_LIST]:
        if not f.exists():
            issues.append(f"MISSING: {f}")

    if issues:
        for issue in issues:
            print(f"  ERROR: {issue}")
        print("\nRun 'prepare' first.")
        sys.exit(1)

    # Check train list format
    n_train = 0
    n_val = 0
    missing_wavs = 0
    empty_phonemes = 0
    speakers = set()

    for list_file, label in [(TRAIN_LIST, "train"), (VAL_LIST, "val")]:
        with open(list_file) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    issues.append(
                        f"{label}:{line_no} — expected 3 fields, got {len(parts)}"
                    )
                    continue

                filename, ipa, speaker = parts
                wav_path = WAVS_DIR / filename

                if not wav_path.exists():
                    missing_wavs += 1

                if not ipa or len(ipa) < 3:
                    empty_phonemes += 1

                speakers.add(speaker)

                if label == "train":
                    n_train += 1
                else:
                    n_val += 1

    print(f"  Train entries : {n_train:,}")
    print(f"  Val entries   : {n_val:,}")
    print(f"  Speakers      : {speakers}")
    print(f"  Missing WAVs  : {missing_wavs}")
    print(f"  Empty phonemes: {empty_phonemes}")

    # Check converted weights
    weights_path = TRAINING_DIR / "kokoro_base.pth"
    if weights_path.exists():
        import torch

        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        print(f"  Base weights  : OK ({list(state.keys())})")
    else:
        print(f"  Base weights  : NOT FOUND (run 'convert-weights')")

    # Check a sample phoneme sequence against Kokoro vocab
    config_path = TRAINING_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        vocab = config["vocab"]

        # Check first 10 train entries for unknown symbols
        unknown_chars = set()
        with open(TRAIN_LIST) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                ipa = line.strip().split("|")[1]
                for ch in ipa:
                    if ch not in vocab:
                        unknown_chars.add(ch)

        if unknown_chars:
            print(
                f"  WARNING: Unknown phoneme chars (will be dropped): {unknown_chars}"
            )
            for ch in unknown_chars:
                print(f"    {repr(ch)} (U+{ord(ch):04X})")
        else:
            print(f"  Phoneme vocab : OK (all symbols in Kokoro vocab)")

    if issues:
        print(f"\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  All checks passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for StyleTTS2 fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Generate train/val lists from dataset")
    subparsers.add_parser(
        "precompute",
        help="Pre-compute mel spectrograms and F0 (optional, saves GPU time)",
    )
    p_convert = subparsers.add_parser(
        "convert-weights", help="Convert Kokoro-82M weights to StyleTTS2 format"
    )
    p_convert.add_argument(
        "--force", action="store_true", help="Regenerate even if output exists"
    )

    subparsers.add_parser(
        "patch-styletts2",
        help="Patch StyleTTS2 text_utils.py and meldataset.py with Kokoro's vocab",
    )

    subparsers.add_parser("verify", help="Verify training data integrity")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare()
    elif args.command == "precompute":
        cmd_precompute()
    elif args.command == "convert-weights":
        cmd_convert_weights(force=args.force)
    elif args.command == "patch-styletts2":
        cmd_patch_styletts2()
    elif args.command == "verify":
        cmd_verify()


if __name__ == "__main__":
    main()
