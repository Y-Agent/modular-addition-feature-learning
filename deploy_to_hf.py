#!/usr/bin/env python3
"""Deploy to Hugging Face Spaces.

Prepends the required HF Space metadata to README.md before uploading,
so the GitHub README stays clean (no YAML frontmatter).

Usage:
    python deploy_to_hf.py
    python deploy_to_hf.py --message "Update app"
"""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

REPO_ID = "y-agent/modular-addition-feature-learning"

HF_FRONTMATTER = """\
---
title: Modular Addition Feature Learning
emoji: "\U0001f522"
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: "6.5.1"
app_file: hf_app/app.py
pinned: false
---
"""

IGNORE_PATTERNS = [
    "trained_models/*", "saved_models/*", "src/saved_models/*",
    ".git/*", ".claude/*", ".DS_Store", "tmp/*",
    "notebooks/*", "figures/*", "__pycache__/*", "src/wandb/*",
    "deploy_to_hf.py",  # no need to upload the deploy script itself
]


def main():
    parser = argparse.ArgumentParser(description="Deploy to HF Spaces")
    parser.add_argument("--message", "-m", default="Update app",
                        help="Commit message for HF")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    readme_path = os.path.join(project_root, "README.md")

    # Read original README
    with open(readme_path, "r") as f:
        original_readme = f.read()

    # Prepend HF frontmatter
    hf_readme = HF_FRONTMATTER + "\n" + original_readme

    # Write the modified README temporarily
    try:
        with open(readme_path, "w") as f:
            f.write(hf_readme)

        print(f"Uploading to {REPO_ID}...")
        api = HfApi()
        api.upload_folder(
            folder_path=project_root,
            repo_id=REPO_ID,
            repo_type="space",
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=args.message,
        )
        print("Done.")
    finally:
        # Restore original README
        with open(readme_path, "w") as f:
            f.write(original_readme)
        print("Restored original README.md")


if __name__ == "__main__":
    main()
