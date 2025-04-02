from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import warnings
from threading import Thread

# 忽略 Flash Attention 警告
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

# 检查是否有 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    r"D:\xiaojun\work\nlp\RAG_code\models\LLMs\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B",
    torch_dtype="auto",
    low_cpu_mem_usage=False  # 禁用 low_cpu_mem_usage
).to(device).eval()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(r"D:\xiaojun\work\nlp\RAG_code\models\LLMs\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B")

# 确保 pad_token_id 和 eos_token_id 正确设置
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 对话循环
while True:
    query = input("\nUser: ")
    if query.lower() == "exit":
        break

    # 构建输入
    messages = [{"role": "user", "content": query}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # 显式设置 attention_mask
    attention_mask = inputs.ne(tokenizer.pad_token_id).long().to(device)

    # 创建流式生成器
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,  # 跳过输入部分
        skip_special_tokens=True  # 跳过特殊 token
    )

    # 生成回复（异步）
    generation_kwargs = {
        "input_ids": inputs,
        "attention_mask": attention_mask,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 512,
        "streamer": streamer
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 流式输出
    print("Assistant: ", end="", flush=True)
    for new_text in streamer:
        print(new_text, end="", flush=True)
    print()  # 换行