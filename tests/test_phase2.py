import tempfile
import unittest
from pathlib import Path

from app.data import ManifestRow, VOCAB_SYMBOLS, make_split, min_ctc_frames, parse_alignment, validate_rows


class Phase2Tests(unittest.TestCase):
    def test_alignment_and_ctc_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.align"
            path.write_text("0\t10\tsil\n10\t20\tPošalji\n20\t30\ta\n30\t40\tsil\n", encoding="utf-8")
            self.assertEqual(parse_alignment(path), "pošalji a")
        self.assertEqual(min_ctc_frames("anna"), 5)

    def test_invalid_silence_timing_does_not_change_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.align"
            path.write_text("0\t10\ta\n20\t15\tsil\n", encoding="utf-8")
            self.assertEqual(parse_alignment(path), "a")

    def test_vocab_has_29_classes(self):
        self.assertEqual(VOCAB_SYMBOLS[0], "<blank>")
        self.assertEqual(len(VOCAB_SYMBOLS), 29)
        self.assertEqual(len(VOCAB_SYMBOLS), len(set(VOCAB_SYMBOLS)))

    def test_speaker_disjoint_split(self):
        rows = []
        for speaker in range(1, 7):
            for sample in range(2):
                rows.append(ManifestRow(f"spk{speaker:02}_{sample}", f"spk{speaker:02}", "v", "a", "a", 1, 25, 25, 1, 1, 1, 1))
        split = make_split(rows, seed=42)
        speaker_sets = [set(split["speakers"][name]) for name in ("train", "validation", "test")]
        self.assertFalse(speaker_sets[0] & speaker_sets[1])
        self.assertFalse(speaker_sets[0] & speaker_sets[2])
        self.assertFalse(speaker_sets[1] & speaker_sets[2])

    def test_unknown_character_is_rejected(self):
        row = ManifestRow("spk01_0", "spk01", "v", "a", "q", 1, 25, 25, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "van vokabulara"):
            validate_rows([row])


if __name__ == "__main__":
    unittest.main()
