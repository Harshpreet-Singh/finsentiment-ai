"""
Financial PhraseBank loader + inspector.

Dataset: takala/financial_phrasebank on HuggingFace
Labels: 0 = negative, 1 = neutral, 2 = positive
"""

from collections import Counter
from datasets import load_dataset

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def load_phrasebank(split: str = "sentences_75agree"):
    """
    Load Financial PhraseBank from HuggingFace.

    Args:
        split: one of 'sentences_allagree', 'sentences_75agree',
               'sentences_66agree', 'sentences_50agree'

    Returns:
        HuggingFace DatasetDict with 'train' split.
    """
    ds = load_dataset("takala/financial_phrasebank", split)
    return ds


def inspect(ds):
    """Print class distribution + sample examples per class."""
    print(f"\n{'='*60}")
    print(f"Total samples: {len(ds['train'])}")
    print(f"{'='*60}\n")

    labels = ds["train"]["label"]
    counts = Counter(labels)

    print("Class distribution:")
    for label_id, name in LABEL_NAMES.items():
        count = counts[label_id]
        pct = 100 * count / len(labels)
        print(f"  {name:>10}: {count:>5} ({pct:5.2f}%)")

    print(f"\n{'='*60}")
    print("Sample examples per class:")
    print(f"{'='*60}\n")

    for label_id, name in LABEL_NAMES.items():
        print(f"--- {name.upper()} ---")
        shown = 0
        for row in ds["train"]:
            if row["label"] == label_id:
                print(f"  • {row['sentence']}")
                shown += 1
                if shown == 3:
                    break
        print()


if __name__ == "__main__":
    ds = load_phrasebank("sentences_75agree")
    inspect(ds)