import random
import json
from datasets import load_dataset

"""
Creates toxicity_bench_sample.json with a small sample of prompts
from the RealToxicityPrompts dataset.

Output format (list of dicts):

[
  {
    "id": "realtoxicityprompts-12345",
    "source": "RealToxicityPrompts",
    "toxicity": 0.87,
    "text": "prompt text here..."
  },
  ...
]
"""

SAMPLE_SIZE = 200  # adjust if you want more or fewer


def main():
    print("Loading RealToxicityPrompts from Hugging Face...")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")  # [web:126]

    # Filter to entries that actually have a prompt
    records = [row for row in ds if row.get("prompt")]
    print(f"Total prompts available: {len(records)}")

    # Shuffle and sample
    random.seed(42)
    random.shuffle(records)
    sample = records[:SAMPLE_SIZE]

    # Build simple list for the app
    out = []
    for i, row in enumerate(sample):
        prompt = row["prompt"]["text"] if isinstance(row["prompt"], dict) else row["prompt"]
        toxicity = row.get("toxicity", {}).get("toxicity", None)  # numeric score if present [web:133]
        out.append(
            {
                "id": f"realtoxicityprompts-{i}",
                "source": "RealToxicityPrompts",
                "toxicity": toxicity,
                "text": prompt,
            }
        )

    out_path = "toxicity_bench_sample.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(out)} prompts to {out_path}")


if __name__ == "__main__":
    main()
