"""Build MATH in the zhuzilin/gsm8k format Miles reads, and optionally push it to a private HF repo.

    .venv/bin/python prepare-math.py                         # writes datasets/math/
    .venv/bin/python prepare-math.py --push blambert/math    # also uploads it, as a private dataset
    .venv/bin/python prepare-math.py --min-level 5 --out datasets/math-l5   # only the hardest problems

Training data is MATH's 7,500-problem train split, with the last \\boxed{} of each solution as the
label; evals use MATH-500, the standard 500-problem subset of its test split. Both come from mirrors
pinned to a commit: the original hendrycks/competition_math was taken down from Hugging Face.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

TRAIN_REPO, TRAIN_REV = "EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3"
TEST_REPO, TEST_REV = "HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]  # fmt: skip
# zhuzilin/gsm8k's system prompt, so the reward finds the answer the same way
SYSTEM = "You are a helpful assistant. Please put the answer within \\boxed{}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent))
    ap.add_argument("--out", type=Path, default=root / "datasets" / "math")
    ap.add_argument("--push", metavar="REPO", help="upload to this private HF dataset repo")
    ap.add_argument("--min-level", type=int, default=1, help="keep only training problems of this level (1-5) and up")
    args = ap.parse_args()

    train, test = load_train(), load_test()
    train = train[train.level >= args.min_level].reset_index(drop=True)
    overlap = set(train.question) & set(test.question)
    assert not overlap, f"{len(overlap)} MATH-500 problems are in the train split"

    args.out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out / "train.parquet")
    test.to_parquet(args.out / "test500.parquet")
    (args.out / "README.md").write_text(card(train, test, args.min_level))
    print(f"wrote {len(train)} train and {len(test)} test problems to {args.out}")
    print(train.groupby(["level"]).size().to_string())

    if args.push:
        push(args.out, args.push)


def load_train() -> pd.DataFrame:
    parts = []
    for s in SUBJECTS:
        f = hf_hub_download(TRAIN_REPO, f"{s}/train-00000-of-00001.parquet", repo_type="dataset", revision=TRAIN_REV)
        parts.append(pd.read_parquet(f))
    raw = pd.concat(parts, ignore_index=True)

    df = pd.DataFrame({
        "question": raw.problem,
        "answer": raw.solution,
        "label": raw.solution.map(label),
        "subject": raw.type,
        "level": raw.level.str.extract(r"(\d)", expand=False).astype("Int64"),  # a few are "Level ?"
    })  # fmt: skip
    missing = df.label.isna()
    print(f"dropped {missing.sum()} train problems without one clear \\boxed{{}} answer in the solution")
    return with_messages(df[~missing].reset_index(drop=True))


def load_test() -> pd.DataFrame:
    f = hf_hub_download(TEST_REPO, "test.jsonl", repo_type="dataset", revision=TEST_REV)
    raw = pd.DataFrame([json.loads(line) for line in open(f)])
    df = pd.DataFrame({
        "question": raw.problem,
        "answer": raw.solution,
        "label": raw.answer,
        "subject": raw.subject,
        "level": raw.level.astype("Int64"),
        "unique_id": raw.unique_id,
    })  # fmt: skip
    return with_messages(df)


def label(solution: str) -> str | None:
    """The solution's \\boxed{} answer, or None if it has none, an empty one or several that differ."""
    # 25 solutions box different values, mostly "find all" answers split across boxes, so the
    # last box alone would be an incomplete label
    found = set(boxes(solution))
    return found.pop() if len(found) == 1 and "" not in found else None


def boxes(text: str) -> list[str]:
    """Contents of every \\boxed{...} and \\fbox{...}, with nested braces."""
    out, i = [], 0
    while (s := min((p for p in (text.find("\\boxed{", i), text.find("\\fbox{", i)) if p >= 0), default=-1)) >= 0:
        j = text.index("{", s) + 1
        depth, k = 1, j
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[j : k - 1].strip())
        i = k
    return out


def with_messages(df: pd.DataFrame) -> pd.DataFrame:
    df["messages"] = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}] for q in df.question]
    return df


def card(train: pd.DataFrame, test: pd.DataFrame, min_level: int) -> str:
    levels = "" if min_level == 1 else f" of level {min_level} and up"
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: test
    path: test500.parquet
---

# MATH for Miles

MATH in the format of [zhuzilin/gsm8k](https://huggingface.co/datasets/zhuzilin/gsm8k), for GRPO with
[Miles](https://github.com/radixark/miles): a chat-format `messages` column with a system prompt asking
for `\\boxed{{}}`, and the answer in `label`. Built by `prepare-math.py` in
[miles-dgx-spark](https://github.com/belambert/miles-dgx-spark).

- **train:** {len(train)} problems{levels} from MATH's train split, via
  [{TRAIN_REPO}](https://huggingface.co/datasets/{TRAIN_REPO}) at `{TRAIN_REV[:10]}`. The label is the
  solution's `\\boxed{{}}` answer; problems whose solution has none, an empty one, or several different
  ones are dropped.
- **test:** {len(test)} problems, MATH-500 from
  [{TEST_REPO}](https://huggingface.co/datasets/{TEST_REPO}) at `{TEST_REV[:10]}`, labeled with its
  `answer` column.

Columns: `question`, `answer` (the reference solution), `label`, `subject`, `level` (1–5), `messages`,
and in test `unique_id`.

The problems come from the MATH dataset (Hendrycks et al., 2021), drawn from AMC, AIME and other
competitions. This copy is private, for personal research use.
"""


def push(out: Path, repo: str) -> None:
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    # create_repo leaves an existing repo's visibility alone, so check it
    assert api.dataset_info(repo).private, f"{repo} exists and is public; not uploading MATH to it"
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=out, commit_message="Build with prepare-math.py")
    print(f"pushed to https://huggingface.co/datasets/{repo} (private)")


if __name__ == "__main__":
    main()
