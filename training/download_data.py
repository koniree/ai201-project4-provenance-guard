"""
Downloads the AI-GA (AI-Generated Abstracts) dataset used to train the
learned ML signal.

Source: https://github.com/panagiotisanagnostou/AI-GA
28,662 samples, perfectly balanced (14,331 human / 14,331 AI-generated).
- label 0 = original human-written abstract, sourced from the CORD-19
  COVID-19 research corpus (https://github.com/allenai/cord19)
- label 1 = AI-generated abstract, generated with GPT-3

Domain note: this dataset is scientific abstracts, not creative writing.
It's used here because it's a real, sizable, cleanly-labeled, balanced
dataset that's actually retrievable in this environment (hosted directly
in a GitHub repo rather than behind a Hugging Face Hub download). See
README.md "Known Limitations" for what this means for the trained signal.
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/panagiotisanagnostou/AI-GA/main/ai-ga-dataset.csv"
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "ai-ga-dataset.csv")


def download():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        print(f"Already downloaded: {OUT_PATH}")
        return
    print(f"Downloading dataset from {URL} ...")
    urllib.request.urlretrieve(URL, OUT_PATH)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Saved {size_mb:.1f} MB to {OUT_PATH}")


if __name__ == "__main__":
    download()
