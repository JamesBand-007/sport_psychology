from openai import OpenAI
import json
from tqdm import tqdm
import requests

def openai_chatgpt_function_stream(question):

    url = "https://ai.comfly.chat/v1"  # 可以替换为任何代理的接口
    OPENAI_API_KEY = "sk-ysb2HIUWIS87ByOKCfA3C2E0Df0c4a4e953006843dF70372"  # openai官网获取key

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=url  # 设置自定义的 API 地址
    )

    # 调用 GPT-4o 模型
    response = client.chat.completions.create(
        # model="gpt-4o",  # 模型名称
        # model="deepseek-r1",
        model= "grok-3",

        messages=[
            {"role": "system", "content": "You are an expert in sports psychology."},
            {"role": "user", "content":  question}
        ],
        stream=True  # 启用流式响应
    )

    # 处理流式响应
    answer = ""
    for chunk in response:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            answer += content

    return answer

def process_questions_from_json(input_json_path, output_json_path):
    # 读取输入的json文件
    with open(input_json_path, 'r', encoding="utf-8") as f:
        question_dict = json.load(f)

    # 获取所有问腿
    questions = list(question_dict.values()) * 3
    # print(questions)

    # 初始化结果列表
    results = []

    # 初始化进度条
    with (tqdm(total=len(questions), desc="Processing Question") as pbar):
        for i, question in enumerate(questions):
            # if i < 297:
            #     pbar.update(1)
            #     continue

            if question:
                # print(f"\nProcessing Question {i + 1}：{ question }")
                answer = openai_chatgpt_function_stream(question)
                # print(f"\n answer: {answer}")
                # 将问题和答案添加到结果列表中
                results.append({"question":question,"answer":answer})

                # 将结果保存至指定json文件中
                with open(output_json_path, 'w', encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

            pbar.update(1) # 更新进度条


    print(f"\n Results saved to {output_json_path}")

if __name__ == "__main__":
    input_json_file = r"D:\xiaojun\work\knowledgeDiscovery\心理(河北石家庄)\2024-09-23after\深入工作\文档\基于大模型的运动心理学应用研究\资料\data\question\测试题zh-100.json"  # 输入的 JSON 文件路径
    output_json_file = r"D:\xiaojun\work\knowledgeDiscovery\心理(河北石家庄)\2024-09-23after\深入工作\文档\基于大模型的运动心理学应用研究\资料\data\answer\grok-3_new.json"  # 输出的 JSON 文件路径
    process_questions_from_json(input_json_file, output_json_file)