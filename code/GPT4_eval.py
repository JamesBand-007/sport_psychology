from openai import OpenAI
import json
from tqdm import tqdm
import requests

def openai_chatgpt_eval_stream(prompt,question,answer):

    url = "https://ai.comfly.chat/v1"  # 可以替换为任何代理的接口
    OPENAI_API_KEY = "sk-ysb2HIUWIS87ByOKCfA3C2E0Df0c4a4e953006843dF70372"  # openai官网获取key

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=url  # 设置自定义的 API 地址
    )
    QA = {
        "question": question,
        "answer": answer
    }
    # 调用 GPT-4o 模型
    response = client.chat.completions.create(
        model="gpt-4o",  # 模型名称
        messages=[
            {"role": "system", "content": "You are an expert in sports psychology."},
            {"role": "user", "content":  f"{prompt}\n{QA}"}
        ],
        stream=True  # 启用流式响应
    )

    # 处理流式响应
    score = ""
    for chunk in response:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            score += content

    return score

def process_data_from_json(prompt_json_file,input_json_file, output_json_file):
    prompt_level1 = ""
    prompt_level2 = ""
    prompt_level3 = ""
    # 读取prompt的json文件
    with open(prompt_json_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
        prompt_level1 = data[0]["prompt"]
        prompt_level2 = data[1]["prompt"]
        prompt_level3 = data[2]["prompt"]

    score_content = []

    # 读取待测数据的json文件
    with open(input_json_file, 'r', encoding="utf-8") as f:
        data = json.load(f)

        # 初始化进度条
        with tqdm(total=len(data), desc="Evaluting QA") as pbar:
            # 循环评测，并保存评测结果
            for i,item in enumerate(data):
                # if  i != 29 and i != 32 and i != 177 and i != 270:
                # # if i < 279:
                #     pbar.update(1)  # 更新进度条
                #     continue
                if item["type"] == "Level1":
                    prompt = prompt_level1
                elif item["type"] == "Level2":
                    prompt = prompt_level2
                else:
                    prompt = prompt_level3

                question = item["question"]
                answer = item["answer"]


                score = openai_chatgpt_eval_stream(prompt, question, answer)

                score_content.append({"score":score})
                # 将结果保存至指定json文件中
                with open(output_json_file, 'w', encoding="utf-8") as f:
                    json.dump(score_content, f, ensure_ascii=False, indent=4)

                pbar.update(1)  # 更新进度条

            # print(questions)

    # 初始化结果列表
    results = []


    print(f"\n Results saved to {output_json_file}")

if __name__ == "__main__":
    input_json_file = '../../../knowledgeDiscovery/心理(河北石家庄)/2024-09-23after/深入工作/文档/基于大模型的运动心理学应用研究/资料/data/eval/vllm_batch_eval/batch_rag_test_32B_results_new.json'  # 输入的 JSON 文件路径
    output_json_file = '../../../knowledgeDiscovery/心理(河北石家庄)/2024-09-23after/深入工作/文档/基于大模型的运动心理学应用研究/资料/data/eval/vllm_batch_eval/score/32B_rag_score.json'  # 输出的 JSON 文件路径
    prompt_json_file = r"D:\xiaojun\work\knowledgeDiscovery\心理(河北石家庄)\2024-09-23after\深入工作\文档\基于大模型的运动心理学应用研究\资料\data\eval\eval_prompt.json"
    process_data_from_json(prompt_json_file, input_json_file, output_json_file)