from huggingface_hub import snapshot_download, login

login("Your Huggingface access token") 


if __name__ == "__main__":
    """
    HF_ENDPOINT=https://hf-mirror.com \
    python -m hf_download
    """
    snapshot_download(
        repo_id="MSALab/LoomVideo",
        repo_type="model",
        ignore_patterns=["assets*", "examples*"],
        local_dir="./checkpoints/LoomVideo",
    )

    print("The End")