from huggingface_hub import snapshot_download

# 下载模型文件并保存到本地
file_path = snapshot_download(repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",  local_dir="E:\tsinghua\OLLAMA_MODEL\manifests\registry.ollama.ai\library\deepseek-coder")

print(f"Model file downloaded to: {file_path}")
