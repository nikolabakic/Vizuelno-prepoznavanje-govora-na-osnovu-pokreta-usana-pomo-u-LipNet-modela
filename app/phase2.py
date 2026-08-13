from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from app.data import (
    ManifestRow,
    build_manifest,
    make_split,
    vocab_payload,
    write_json,
    write_manifest,
)
from app.preprocessing import build_rois, write_qa_sheet, write_rois


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            ManifestRow(
                **{
                    **row,
                    "duration_s": float(row["duration_s"]),
                    "num_frames": int(row["num_frames"]),
                    "fps": float(row["fps"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "target_length": int(row["target_length"]),
                    "min_ctc_frames": int(row["min_ctc_frames"]),
                }
            )
            for row in csv.DictReader(handle)
        ]


def build_command(args: argparse.Namespace) -> None:
    rows = build_manifest(args.corpus)
    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, args.output / "manifest.csv")
    split = make_split(rows, seed=args.seed)
    write_json(split, args.output / "split.json")
    write_json(vocab_payload(), args.output / "vocab.json")
    counts = {name: len(split[name]) for name in ("train", "validation", "test")}
    print(
        f"Manifest: {len(rows)} primera, {len({row.speaker_id for row in rows})} govornika"
    )
    print(f"Split: {counts}; izlaz: {args.output}")


def roi_command(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.input / "manifest.csv")
    rois = build_rois(manifest, args.corpus.resolve(), samples=args.samples)
    write_rois(rois, args.input / "roi.csv")
    qa_ids = write_qa_sheet(
        manifest, rois, args.corpus.resolve(), args.input / "roi_qa.jpg", args.qa_count
    )
    write_json(
        {"sample_ids": qa_ids, "manual_review_complete": False},
        args.input / "roi_qa.json",
    )
    fallback = sum(row.source != "detected" for row in rois)
    print(
        f"ROI: {len(rois)} primera, fallback={fallback}; ručno pregledaj {args.input / 'roi_qa.jpg'}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="LipNet faza 2: manifest, split, vokabular i mouth ROI"
    )
    commands = result.add_subparsers(required=True)
    build = commands.add_parser(
        "build", help="Generiši i validiraj manifest, split i vokabular"
    )
    build.add_argument("--corpus", type=Path, default=Path("processed/processed"))
    build.add_argument("--output", type=Path, default=Path("artifacts/phase2"))
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(handler=build_command)
    roi = commands.add_parser(
        "roi", help="Detektuj stabilan mouth ROI i napravi QA sliku"
    )
    roi.add_argument("--corpus", type=Path, default=Path("processed/processed"))
    roi.add_argument("--input", type=Path, default=Path("artifacts/phase2"))
    roi.add_argument("--samples", type=int, default=5)
    roi.add_argument("--qa-count", type=int, default=30)
    roi.set_defaults(handler=roi_command)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.handler(arguments)
