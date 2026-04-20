# upload_local_model_to_hf.py
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


# ====== 你改这里 ======
LOCAL_MODEL_DIR = "output/Qwen3-1.7B_openthoughts_sft_step198"   # 本地模型目录
REPO_ID = "zsqzz/Qwen3-1.7B_openthoughts_sft_step198"    # 例如: oliverzhu/qwen3-1.7b-myft
HF_TOKEN = os.environ.get("HF_TOKEN")           # 也可以直接写成字符串，但不推荐
PRIVATE = False                                  # 是否私有仓库
COMMIT_MESSAGE = "Upload local model"
# =====================


def main():
    local_dir = Path(LOCAL_MODEL_DIR)
    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Local model dir not found: {local_dir}")

    if HF_TOKEN is None:
        raise ValueError(
            "HF_TOKEN is not set. Please run: export HF_TOKEN=your_token"
        )

    required_candidates = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    existing = [x for x in required_candidates if (local_dir / x).exists()]
    print(f"Found optional/common files: {existing}")

    # 创建仓库。如果已存在，不报错
    create_repo(
        repo_id=REPO_ID,
        token=HF_TOKEN,
        repo_type="model",
        private=PRIVATE,
        exist_ok=True,
    )

    api = HfApi(token=HF_TOKEN)

    # 上传整个目录
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=COMMIT_MESSAGE,
        # 可按需忽略
        ignore_patterns=[
            "*.tmp",
            "*.log",
            ".DS_Store",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "rng_state.pth",
            "events.out.tfevents.*",
            "__pycache__/*",
        ],
    )

    print(f"Done! Uploaded to: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()