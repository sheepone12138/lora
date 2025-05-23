"""优化GPU显存使用"""
import torch
import os
torch.cuda.empty_cache()  # 清空显存，释放 GPU 资源

"""配置"""
output_path = r"E:/tsinghua/OLLAMA_MODEL/manifests/registry.ollama.ai/library/lora"  # 输出路径
model_path = r"E:/tsinghua/OLLAMA_MODEL/manifests/registry.ollama.ai/library/deepseek-coder"  # 模型路径
data_path = r"E:/tsinghua/DeepSeek-Coder-main/LoRa/data/filtered_instructions.json"  # 数据路径

"""加载模型和分词器"""
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,        # 防止中文数据切分异常
    trust_remote_code=True,
    padding_side="right",   # 保证中文对齐时表现更稳定
    local_files_only=True  # 使用本地模型
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,  # 使用本地模型
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 节省显存
    ## device_map="auto"        # 根据显存自动分配
)

# 关闭缓存并开启梯度检查点，减少显存消耗
model.config.use_cache = False
model.gradient_checkpointing_enable()


"""数据加载和预处理"""
import pandas as pd
from datasets import Dataset

# 加载数据
df = pd.read_json(data_path)  # 读取 JSON 文件
ds = Dataset.from_pandas(df)

"""数据预处理"""
def process_func(example):
    MAX_LENGTH = 384  
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(
        f"<|im_start|>system\n你是一个编程助手，负责编写opendrive相关xml文件<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False
    )

    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 数据映射
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

"""配置LoRa训练参数"""
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,      # 指定模型类型（例如：因果语言模型 CLM）
    target_modules=["q_proj", "v_proj"], # 指定哪些模块要进行 LoRA 微调
    inference_mode=False,              # 设置是否仅用于推理
    r=4,                               # LoRA 的秩 (rank)
    lora_alpha=16,                     # LoRA 的缩放参数
    lora_dropout=0.05,                 # LoRA 的 Dropout 防止过拟合
    bias="none"                        # 偏置参数设置
)
model = get_peft_model(model, config)
model.print_trainable_parameters()     # 输出可训练参数数目

"""训练参数配置"""
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir=output_path,  # 输出模型保存路径
    per_device_train_batch_size=2,             # 每个 GPU 的 batch_size
    gradient_accumulation_steps=16,            # 梯度累积步数
    gradient_checkpointing=True,               # 启用梯度检查点，节省显存
    bf16=True,                                 # 使用 `bfloat16` 数值表示（减少显存占用）
    learning_rate=2e-5,                        # 学习率，控制模型学习的速度
    num_train_epochs=3,                        # 训练的总轮数
    logging_steps=10,                          # 每 10 步输出一次日志
    save_strategy="steps",                     # 每隔几步保存模型
    save_steps=100,                            # 每 100 步保存一次
    optim="adamw_torch_fused",                 # 使用 `adamw_torch_fused` 提高训练速度
    max_grad_norm=0.3                          # 梯度裁剪，防止梯度爆炸
)

"""训练模型"""
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds
)

trainer.train()


"""模型保存"""
model.save_pretrained(os.path.join(output_path,"final"))  # 保存模型
