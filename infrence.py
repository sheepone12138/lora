from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch
from peft import PeftModel


model_path = r"E:/tsinghua/OLLAMA_MODEL/manifests/registry.ollama.ai/library/deepseek-coder"  # 模型路径
lora_path = r"E:/tsinghua/OLLAMA_MODEL/manifests/registry.ollama.ai/library/lora"  # 数据路径

def load_model(model_path, lora_path):
    """加载模型和分词器"""

    # 配置4比特量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("正在加载模型和分词器...")
    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,        # 防止中文数据切分异常
            trust_remote_code=True,
            padding_side="right",  # 保证中文对齐时表现更稳定
            local_files_only=True  # 使用本地模型
        )
    except Exception as e:
        print(f"加载分词器失败: {e}")
        raise
    
    # 加载基础模型并应用量化配置
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=quantization_config,  # 应用量化配置
            device_map="auto",       # 根据显存自动分配
            torch_dtype=torch.bfloat16,  # 使用bfloat16节省显存
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise

#    关闭缓存并开启梯度检查点，减少显存消耗
#    model.config.use_cache = False  # 推理变慢，但是可以优化显存

#    """加载LoRA权重"""
#    print("正在加载LoRA权重...")
#    model = PeftModel.from_pretrained(model, lora_path)

    return model, tokenizer

def build_history(conversation_history, input_text):
    """构建历史对话记录"""
    prompt = ""
    for user_msg, model_msg in conversation_history:
        prompt += f"Instruction: 继续对话\nInput: {user_msg}\nOutput: {model_msg}\n\n"
    
    # 添加当前轮次
    prompt += f"Instruction: 继续对话\nInput: {input_text}\nOutput: "
    return prompt

def generate_response(model, tokenizer, history):
    """生成回复"""

    prompt = history
        # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    # 移除token_type_ids
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
    inputs = inputs.to(model.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=10240,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0, input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response

def main(max_history=5):
    """
    主函数
    交互式对话
    """
    # 加载模型和分词器
    model, tokenizer = load_model(model_path, lora_path)
    conversation_history = []

    # 交互式对话
    while True:
        input_text = input("请输入文本（输入'quit'退出，输入'new'更新历史对话记录）：").strip()
        if input_text.lower() == 'quit':
            break
        elif input_text.lower() == 'new':
            conversation_history = []
            print("历史对话记录已清空！")
            continue

        history = build_history(conversation_history, input_text)
        response = generate_response(model, tokenizer, history)
        conversation_history.append((input_text, response))
        # 限制历史长度，避免超出模型最大上下文
        if len(conversation_history) > max_history:
            conversation_history.pop(0)  # 移除最早的对话
        print(f"生成的回复：{response}")
        print("===========================")
        print(f"历史记录：{history}")
        print("===========================")

if __name__ == "__main__":
    main()