#!/usr/bin/env python3
"""
Kokoro German Training Dataset Pipeline
========================================
Processes cached Polly MP3s into a clean German TTS training dataset.

Usage:
    # Step 1: Transcribe all MP3s with mlx-whisper (resumable, run overnight)
    uv run python scripts/prepare_dataset.py transcribe

    # Step 2: Filter by language, duration, quality
    uv run python scripts/prepare_dataset.py filter

    # Step 3: Cluster speakers (find distinct Polly voices)
    uv run python scripts/prepare_dataset.py cluster

    # Step 4: Convert audio + generate IPA + write final dataset
    uv run python scripts/prepare_dataset.py format

    # Quick sanity check on a few files before committing to the full run
    uv run python scripts/prepare_dataset.py transcribe --sample 20

    # Print stats at any stage
    uv run python scripts/prepare_dataset.py stats
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("./cache")
DATASET_DIR = Path("./dataset")
AUDIO_DIR = DATASET_DIR / "audio"
TRANSCRIPTIONS_FILE = DATASET_DIR / "transcriptions.jsonl"
FILTERED_FILE = DATASET_DIR / "filtered.jsonl"
SPEAKERS_FILE = DATASET_DIR / "speakers.jsonl"
METADATA_FILE = DATASET_DIR / "metadata.csv"
PHONEMES_FILE = DATASET_DIR / "phonemes.csv"
STATS_FILE = DATASET_DIR / "stats.json"

# ── Filtering thresholds ─────────────────────────────────────────────────────

MIN_DURATION_S = 2.0
MAX_DURATION_S = 30.0
MIN_AVG_LOGPROB = -1.0  # Whisper confidence: closer to 0 is better
MAX_NO_SPEECH_PROB = 0.5  # Reject segments that are likely silence/noise
MIN_WORDS = 3  # Minimum words in transcription
TARGET_LANGUAGE = "de"  # ISO 639-1 German

# ── Whisper model ─────────────────────────────────────────────────────────────

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

# ── Speaker clustering ───────────────────────────────────────────────────────

N_SPEAKER_CLUSTERS = None  # None = auto-detect via DBSCAN


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Transcribe
# ─────────────────────────────────────────────────────────────────────────────


def cmd_transcribe(sample: int | None):
    """Transcribe all MP3s using mlx-whisper. Resumable."""
    import mlx_whisper

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-processed hashes for resumability
    done = set()
    if TRANSCRIPTIONS_FILE.exists():
        with open(TRANSCRIPTIONS_FILE) as f:
            for line in f:
                entry = json.loads(line)
                done.add(entry["hash"])

    # Collect all MP3 files
    all_files = sorted(CACHE_DIR.glob("*.mp3"))
    if sample:
        all_files = all_files[:sample]

    pending = [f for f in all_files if f.stem not in done]
    print(
        f"Total MP3s: {len(all_files)}  |  Already done: {len(done)}  |  Pending: {len(pending)}"
    )

    if not pending:
        print("Nothing to do.")
        return

    print(f"Loading model: {WHISPER_MODEL}")

    errors = 0
    with open(TRANSCRIPTIONS_FILE, "a") as out:
        for mp3 in tqdm(pending, unit="file", desc="Transcribing"):
            try:
                result = mlx_whisper.transcribe(
                    str(mp3),
                    path_or_hf_repo=WHISPER_MODEL,
                    language=None,  # auto-detect
                    task="transcribe",
                    word_timestamps=False,
                    fp16=True,
                )
                # Aggregate segment-level stats
                segments = result.get("segments", [])
                avg_logprob = (
                    sum(s["avg_logprob"] for s in segments) / len(segments)
                    if segments
                    else -9.0
                )
                no_speech_prob = (
                    sum(s["no_speech_prob"] for s in segments) / len(segments)
                    if segments
                    else 1.0
                )
                # Get duration from ffprobe (fast, doesn't re-decode)
                duration = _get_duration(mp3)

                entry = {
                    "hash": mp3.stem,
                    "path": str(mp3),
                    "duration": round(duration, 3),
                    "language": result.get("language", "unknown"),
                    "text": result.get("text", "").strip(),
                    "avg_logprob": round(avg_logprob, 4),
                    "no_speech_prob": round(no_speech_prob, 4),
                    "n_segments": len(segments),
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out.flush()

            except Exception as e:
                tqdm.write(f"ERROR {mp3.name}: {e}")
                errors += 1

    print(f"\nDone. Errors: {errors}")
    _print_transcription_stats()


def _get_duration(path: Path) -> float:
    """Fast duration extraction via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _print_transcription_stats():
    if not TRANSCRIPTIONS_FILE.exists():
        return
    total = done_de = done_en = done_other = 0
    total_duration = 0.0
    with open(TRANSCRIPTIONS_FILE) as f:
        for line in f:
            e = json.loads(line)
            total += 1
            total_duration += e.get("duration", 0)
            lang = e.get("language", "")
            if lang == "de":
                done_de += 1
            elif lang == "en":
                done_en += 1
            else:
                done_other += 1
    print(f"\nTranscription stats:")
    print(f"  Total files   : {total:,}")
    print(f"  Total duration: {total_duration / 3600:.1f}h")
    print(f"  German (de)   : {done_de:,}  ({done_de / total * 100:.1f}%)")
    print(f"  English (en)  : {done_en:,}  ({done_en / total * 100:.1f}%)")
    print(f"  Other         : {done_other:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Filter
# ─────────────────────────────────────────────────────────────────────────────


def cmd_filter():
    """Filter transcriptions by language, duration, and quality."""
    if not TRANSCRIPTIONS_FILE.exists():
        print("No transcriptions.jsonl found. Run: transcribe first.")
        sys.exit(1)

    entries = []
    with open(TRANSCRIPTIONS_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Input: {len(entries):,} entries")

    reasons = {
        "wrong_language": 0,
        "too_short": 0,
        "too_long": 0,
        "low_confidence": 0,
        "high_no_speech": 0,
        "too_few_words": 0,
    }
    kept = []

    for e in entries:
        if e.get("language") != TARGET_LANGUAGE:
            reasons["wrong_language"] += 1
            continue
        dur = e.get("duration", 0)
        if dur < MIN_DURATION_S:
            reasons["too_short"] += 1
            continue
        if dur > MAX_DURATION_S:
            reasons["too_long"] += 1
            continue
        if e.get("avg_logprob", -9.0) < MIN_AVG_LOGPROB:
            reasons["low_confidence"] += 1
            continue
        if e.get("no_speech_prob", 1.0) > MAX_NO_SPEECH_PROB:
            reasons["high_no_speech"] += 1
            continue
        text = e.get("text", "")
        if len(text.split()) < MIN_WORDS:
            reasons["too_few_words"] += 1
            continue
        kept.append(e)

    total_duration = sum(e.get("duration", 0) for e in kept)
    print(f"\nKept: {len(kept):,}  ({total_duration / 3600:.1f}h of audio)")
    print(f"Dropped:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        if count:
            print(f"  {reason:<20}: {count:,}")

    with open(FILTERED_FILE, "w") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nWrote {FILTERED_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Cluster speakers
# ─────────────────────────────────────────────────────────────────────────────


def cmd_cluster():
    """Extract speaker embeddings and cluster into distinct Polly voices."""
    if not FILTERED_FILE.exists():
        print("No filtered.jsonl found. Run: filter first.")
        sys.exit(1)

    import numpy as np

    entries = []
    with open(FILTERED_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Loading {len(entries):,} filtered entries for speaker clustering...")

    # Use a random sample for embedding (full set can be slow)
    # We sample up to 3000 files to build the clustering model
    MAX_EMBED = 3000
    rng = __import__("random").Random(42)
    sample = rng.sample(entries, min(MAX_EMBED, len(entries)))

    print(f"Extracting speaker embeddings from {len(sample):,} files...")
    print("Loading resemblyzer VoiceEncoder...")

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        encoder = VoiceEncoder()
    except Exception as e:
        print(f"Failed to load resemblyzer: {e}")
        print('Falling back to single-speaker labeling (all files -> "d_speaker0")')
        _write_speakers_single(entries)
        return

    embeddings = []
    valid_entries = []
    errors = 0

    for entry in tqdm(sample, desc="Embedding"):
        try:
            wav = preprocess_wav(entry["path"])
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
            valid_entries.append(entry)
        except Exception as e:
            errors += 1

    if errors:
        print(f"Embedding errors: {errors}")

    if not embeddings:
        print("No embeddings extracted. Falling back to single-speaker.")
        _write_speakers_single(entries)
        return

    embeddings = np.array(embeddings)
    # resemblyzer embeddings are already L2-normalized; just ensure array is float32
    embeddings = embeddings.astype(np.float32)

    print(f"Clustering {len(embeddings)} embeddings...")
    n_speakers, labels = _cluster_embeddings(embeddings)
    print(f"Detected {n_speakers} distinct speaker(s)")

    # Build a lookup from file path to speaker label using the sample
    path_to_speaker: dict[str, str] = {}
    for entry, label in zip(valid_entries, labels):
        speaker_id = f"d_speaker{label}"
        path_to_speaker[entry["path"]] = speaker_id

    # For entries NOT in the sample, find nearest speaker using centroid
    # Build centroids per speaker
    centroids = {}
    for label in range(n_speakers):
        mask = labels == label
        centroids[label] = embeddings[mask].mean(axis=0)

    # Assign all entries (not just sample) by re-embedding or nearest centroid
    print(f"Assigning speakers to all {len(entries):,} files...")
    print("(For unsampled files, we embed and find nearest centroid)")

    unsampled = [e for e in entries if e["path"] not in path_to_speaker]
    centroid_matrix = np.array([centroids[i] for i in range(n_speakers)])

    for entry in tqdm(unsampled, desc="Assigning"):
        try:
            wav = preprocess_wav(entry["path"])
            emb = encoder.embed_utterance(wav)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            dists = np.dot(centroid_matrix, emb)
            best = int(np.argmax(dists))
            path_to_speaker[entry["path"]] = f"d_speaker{best}"
        except Exception:
            path_to_speaker[entry["path"]] = "d_speaker0"

    # Write speakers.jsonl
    with open(SPEAKERS_FILE, "w") as f:
        for entry in entries:
            entry["speaker"] = path_to_speaker.get(entry["path"], "d_speaker0")
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Print speaker distribution
    from collections import Counter

    speaker_counts = Counter(path_to_speaker.values())
    print("\nSpeaker distribution:")
    for spk, count in sorted(speaker_counts.items()):
        dur = sum(
            e["duration"] for e in entries if path_to_speaker.get(e["path"]) == spk
        )
        print(f"  {spk}: {count:,} files  ({dur / 3600:.1f}h)")

    print(f"\nWrote {SPEAKERS_FILE}")
    print("\nNext: manually listen to a sample from each speaker to verify gender,")
    print("then rename speakers to Kokoro convention: df_name (female), dm_name (male)")
    print(
        "Edit speakers.jsonl or re-run with --rename-speakers dm_speaker0=dm_hans etc."
    )


def _cluster_embeddings(embeddings):
    """Auto-cluster speaker embeddings. Returns (n_speakers, labels array)."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    # Try 1..8 clusters, pick best silhouette score
    best_score = -1
    best_n = 1
    best_labels = None

    max_clusters = min(8, len(embeddings) // 10)

    if max_clusters < 2:
        return 1, np.zeros(len(embeddings), dtype=int)

    for n in range(2, max_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n)
        labels = clustering.fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_n = n
            best_labels = labels

    if best_labels is None:
        return 1, __import__("numpy").zeros(len(embeddings), dtype=int)

    print(f"Best cluster count: {best_n} (silhouette score: {best_score:.3f})")
    return best_n, best_labels


def _write_speakers_single(entries):
    """Fallback: label all entries as a single speaker."""
    with open(SPEAKERS_FILE, "w") as f:
        for entry in entries:
            entry["speaker"] = "d_speaker0"
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {SPEAKERS_FILE} (single speaker fallback)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b: Drop speakers
# ─────────────────────────────────────────────────────────────────────────────


def cmd_drop(speakers_to_drop: list[str]):
    """Remove entries for specific speaker IDs from speakers.jsonl."""
    if not SPEAKERS_FILE.exists():
        print("No speakers.jsonl found. Run: cluster first.")
        sys.exit(1)

    entries = []
    with open(SPEAKERS_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Input: {len(entries):,} entries")
    print(f"Dropping speakers: {speakers_to_drop}")

    kept = [e for e in entries if e.get("speaker") not in speakers_to_drop]
    dropped = len(entries) - len(kept)
    dropped_duration = sum(
        e["duration"] for e in entries if e.get("speaker") in speakers_to_drop
    )
    kept_duration = sum(e["duration"] for e in kept)

    print(f"Dropped: {dropped:,} files  ({dropped_duration / 3600:.1f}h)")
    print(f"Kept   : {len(kept):,} files  ({kept_duration / 3600:.1f}h)")

    with open(SPEAKERS_FILE, "w") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nUpdated {SPEAKERS_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Format (convert audio + IPA + write final dataset)
# ─────────────────────────────────────────────────────────────────────────────


def cmd_format(rename_speakers: list[str] | None):
    """Convert MP3→WAV, generate IPA phonemes, write final dataset."""
    if not SPEAKERS_FILE.exists():
        print("No speakers.jsonl found. Run: cluster first.")
        sys.exit(1)

    entries = []
    with open(SPEAKERS_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    # Apply any manual speaker renames (e.g. d_speaker0=df_anna)
    rename_map = {}
    if rename_speakers:
        for pair in rename_speakers:
            old, new = pair.split("=", 1)
            rename_map[old.strip()] = new.strip()
    if rename_map:
        print(f"Applying speaker renames: {rename_map}")
        for e in entries:
            e["speaker"] = rename_map.get(e["speaker"], e["speaker"])

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Create per-speaker subdirectories
    speakers = sorted(set(e["speaker"] for e in entries))
    for spk in speakers:
        (AUDIO_DIR / spk).mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(entries):,} MP3s to 24kHz mono WAV...")
    errors = 0
    skipped = 0
    converted = 0

    for entry in tqdm(entries, desc="Converting"):
        spk = entry["speaker"]
        wav_path = AUDIO_DIR / spk / f"{entry['hash']}.wav"

        if wav_path.exists():
            skipped += 1
            entry["wav_path"] = str(wav_path)
            continue

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                entry["path"],
                "-ac",
                "1",
                "-ar",
                "24000",
                "-sample_fmt",
                "s16",
                str(wav_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            errors += 1
            continue

        entry["wav_path"] = str(wav_path)
        converted += 1

    print(f"Converted: {converted}  Skipped (exists): {skipped}  Errors: {errors}")

    # Generate IPA phonemes
    print("Generating IPA phonemes via misaki (espeak-ng German G2P)...")
    from misaki import espeak

    g2p = espeak.EspeakG2P(language="de")

    # Phoneme fixup: ʏ (U+028F, short ü) is not in Kokoro's 178-token vocab.
    # Map it to y (U+0079, long ü) — the duration difference is learned from audio.
    PHONEME_FIXUPS = {
        "\u028f": "y",  # ʏ → y (near-close near-front rounded → close front rounded)
    }

    metadata_rows = []
    phoneme_rows = []
    ipa_errors = 0

    for entry in tqdm(entries, desc="G2P"):
        if "wav_path" not in entry:
            continue
        text = entry["text"]
        spk = entry["speaker"]
        wav_name = f"{entry['speaker']}/{entry['hash']}.wav"

        try:
            phonemes, _ = g2p(text)
            for old, new in PHONEME_FIXUPS.items():
                phonemes = phonemes.replace(old, new)
        except Exception as e:
            ipa_errors += 1
            phonemes = ""

        metadata_rows.append(f"{wav_name}|{text}|{spk}")
        phoneme_rows.append(f"{wav_name}|{phonemes}")

    if ipa_errors:
        print(f"IPA generation errors: {ipa_errors}")

    # Write outputs
    with open(METADATA_FILE, "w") as f:
        f.write("filename|text|speaker\n")
        f.write("\n".join(metadata_rows) + "\n")

    with open(PHONEMES_FILE, "w") as f:
        f.write("filename|ipa\n")
        f.write("\n".join(phoneme_rows) + "\n")

    # Write stats
    total_duration = sum(e["duration"] for e in entries if "wav_path" in e)
    speaker_stats = {}
    for e in entries:
        if "wav_path" not in e:
            continue
        spk = e["speaker"]
        if spk not in speaker_stats:
            speaker_stats[spk] = {"files": 0, "duration_s": 0.0}
        speaker_stats[spk]["files"] += 1
        speaker_stats[spk]["duration_s"] += e["duration"]

    stats = {
        "total_files": len(metadata_rows),
        "total_duration_h": round(total_duration / 3600, 2),
        "speakers": {
            spk: {
                "files": v["files"],
                "duration_h": round(v["duration_s"] / 3600, 2),
            }
            for spk, v in sorted(speaker_stats.items())
        },
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset ready:")
    print(f"  Files   : {stats['total_files']:,}")
    print(f"  Duration: {stats['total_duration_h']}h")
    print(f"  Speakers: {len(speaker_stats)}")
    print(f"  metadata.csv  -> {METADATA_FILE}")
    print(f"  phonemes.csv  -> {PHONEMES_FILE}")
    print(f"  stats.json    -> {STATS_FILE}")
    print(f"  audio/        -> {AUDIO_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────


def cmd_stats():
    """Print statistics for each stage that has been completed."""
    if TRANSCRIPTIONS_FILE.exists():
        print("=== Transcriptions ===")
        _print_transcription_stats()

    if FILTERED_FILE.exists():
        entries = [json.loads(l) for l in open(FILTERED_FILE)]
        dur = sum(e.get("duration", 0) for e in entries)
        print(f"\n=== Filtered ===")
        print(f"  Files   : {len(entries):,}")
        print(f"  Duration: {dur / 3600:.1f}h")

    if SPEAKERS_FILE.exists():
        from collections import Counter

        entries = [json.loads(l) for l in open(SPEAKERS_FILE)]
        counts = Counter(e.get("speaker", "?") for e in entries)
        print(f"\n=== Speakers ===")
        for spk, cnt in sorted(counts.items()):
            dur = sum(e["duration"] for e in entries if e.get("speaker") == spk)
            print(f"  {spk}: {cnt:,} files  ({dur / 3600:.1f}h)")

    if STATS_FILE.exists():
        stats = json.load(open(STATS_FILE))
        print(f"\n=== Final Dataset ===")
        print(f"  Files   : {stats['total_files']:,}")
        print(f"  Duration: {stats['total_duration_h']}h")
        for spk, v in stats.get("speakers", {}).items():
            print(f"  {spk}: {v['files']:,} files  ({v['duration_h']}h)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro German TTS training dataset pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # transcribe
    p_transcribe = subparsers.add_parser(
        "transcribe", help="Transcribe MP3s with mlx-whisper"
    )
    p_transcribe.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Only process first N files (for testing)",
    )
    p_transcribe.add_argument(
        "--model",
        default=WHISPER_MODEL,
        help=f"mlx-whisper model (default: {WHISPER_MODEL})",
    )

    # filter
    subparsers.add_parser("filter", help="Filter by language, duration, quality")

    # cluster
    subparsers.add_parser(
        "cluster", help="Cluster speakers using ECAPA-TDNN embeddings"
    )

    # drop
    p_drop = subparsers.add_parser(
        "drop", help="Remove entries for specific speaker IDs from speakers.jsonl"
    )
    p_drop.add_argument(
        "speakers",
        nargs="+",
        metavar="SPEAKER_ID",
        help="Speaker IDs to drop (e.g. d_speaker0)",
    )

    # format
    p_format = subparsers.add_parser(
        "format", help="Convert audio, generate IPA, write final dataset"
    )
    p_format.add_argument(
        "--rename-speakers",
        nargs="+",
        metavar="OLD=NEW",
        help="Rename speaker IDs (e.g. d_speaker0=df_anna d_speaker1=dm_hans)",
    )

    # stats
    subparsers.add_parser("stats", help="Print statistics for completed stages")

    args = parser.parse_args()

    if args.command == "transcribe":
        cmd_transcribe(sample=args.sample)
    elif args.command == "filter":
        cmd_filter()
    elif args.command == "cluster":
        cmd_cluster()
    elif args.command == "drop":
        cmd_drop(speakers_to_drop=args.speakers)
    elif args.command == "format":
        cmd_format(rename_speakers=args.rename_speakers)
    elif args.command == "stats":
        cmd_stats()


if __name__ == "__main__":
    main()
