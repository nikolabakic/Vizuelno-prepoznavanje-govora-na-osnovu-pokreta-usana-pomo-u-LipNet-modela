"""Speaker-disjoint AI-SPEAK split used by phases 3 and 4.

The supplied subset contains ``spk01`` through ``spk22``.  These lists are the
versioned result of the earlier deterministic seed-42 speaker shuffle.  The
Dataset still discovers sample folders dynamically; no per-sample manifest is
read or generated.
"""

TRAIN_SPEAKERS = (
    "spk01", "spk02", "spk04", "spk05", "spk07", "spk08", "spk09", "spk11",
    "spk12", "spk14", "spk15", "spk17", "spk18", "spk19", "spk20", "spk21",
)
VALIDATION_SPEAKERS = ("spk10", "spk13", "spk16")
TEST_SPEAKERS = ("spk03", "spk06", "spk22")

SPLITS = {
    "train": TRAIN_SPEAKERS,
    "validation": VALIDATION_SPEAKERS,
    "test": TEST_SPEAKERS,
}


def validate_splits() -> None:
    names = tuple(SPLITS)
    for index, left_name in enumerate(names):
        left = set(SPLITS[left_name])
        for right_name in names[index + 1 :]:
            overlap = left & set(SPLITS[right_name])
            if overlap:
                raise ValueError(f"Speaker leakage {left_name}/{right_name}: {sorted(overlap)}")


validate_splits()
