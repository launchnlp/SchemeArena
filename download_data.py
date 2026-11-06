"""Download the SchemeArena scenarios from Hugging Face into ./SchemeArena/.

The dataset is hosted at https://huggingface.co/datasets/launch/SchemeArena.
This fetches SchemeArena.zip and extracts it to ./SchemeArena/example_{id}/,
the layout run_agents.py reads.

Usage:
    pip install huggingface_hub
    python download_data.py
"""
import os
import zipfile

from huggingface_hub import hf_hub_download

REPO_ID = "launch/SchemeArena"
TARGET_DIR = "SchemeArena"


def main():
    if os.path.isdir(TARGET_DIR):
        print(f"./{TARGET_DIR}/ already exists; delete it to download again.")
        return
    zip_path = hf_hub_download(repo_id=REPO_ID, filename="SchemeArena.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(".")
    n = sum(1 for name in os.listdir(TARGET_DIR) if name.startswith("example_"))
    print(f"Extracted {n} scenarios to ./{TARGET_DIR}/")


if __name__ == "__main__":
    main()
