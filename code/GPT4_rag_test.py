from openai import OpenAI
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import requests
import RAG_example as rag

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
    questions = list(question_dict.values())
    # questions = list(question_dict.values())
    # print(questions)

    # 初始化结果列表
    results = []

    # 加载知识库
    embedding_model = SentenceTransformer(r"D:\xiaojun\work\nlp\RAG_code\models\embedding_model\BAAI\bge-large-zh")
    index_file = "../data/knowledge/sport_psychology_1.index"
    docs_file = "../data/knowledge/sport_psychology_1.pkl"
    vector_store = rag.VectorStore(embedding_model, index_file=index_file, docs_file=docs_file)

    count = 0

    # 初始化进度条
    with (tqdm(total=len(questions), desc="Processing Question") as pbar):

        for i, question in enumerate(questions):
            # if i < 64:
            #     pbar.update(1)
            #     continue

            if question:
                # 检索相关文档（top3）
                retrieved_docs = vector_store.search(question, top_k=10)
                # print(f"{i}:{retrieved_docs}")

                prompt = f"请回答问题：{question}\n\n"
                # prompt = "希望以下知识可以帮助你回答问题：\n"

                docs = []
                for idx, doc in enumerate(retrieved_docs, start=1):
                    # 判断文档相关性，过滤低于80%的文档
                    # print(doc[1])
                    similarity = 1 / (1 + doc[1])

                    if similarity >= 0.75:
                        # print(idx)
                        docs.append(doc[0])
                if len(docs):
                    count += 1
                    prompt += "以下知识仅供参考，如果对于问题没有帮助必须忽略知识：\n"

                    for idx, doc in enumerate(docs, start=1):
                        prompt += f"[知识{idx}]: {doc}\n"
                    prompt += f"\n重申问题：{question}\n\n请回答："

                # jsonl_data = {"custom_id": "request-" + str(i + 1), "method": "POST", "url": "/v1/chat/completions", "body": {"model": "/root/zky/models/DeepSeek-R1-Distill-Qwen-32B", "messages": [{"role": "system", "content": "You are an expert in sport and psychology."},{"role": "user", "content": prompt}],"max_completion_tokens": 4096}}
                # with open("../data/vllm_eval/batch_rag_test_32B_80.jsonl", 'a', encoding="utf-8") as jf:
                #     jf.write(json.dumps(jsonl_data, ensure_ascii=False) + '\n')

                print(f"\nProcessing Question {i + 1}：{ prompt }")
                # answer = openai_chatgpt_function_stream(prompt)
                # # answer = "123"
                # # print(f"\n answer: {answer}")
                # # 将问题和答案添加到结果列表中
                # results.append({"question":question,"answer":answer})
                #
                # # 将结果保存至指定json文件中
                # with open(output_json_path, 'w', encoding="utf-8") as f:
                #     json.dump(results, f, ensure_ascii=False, indent=4)

            pbar.update(1) # 更新进度条


    print("count:",count)
    print(f"\n Results saved to {output_json_path}")

if __name__ == "__main__":
    input_json_file = r"D:\xiaojun\work\knowledgeDiscovery\心理(河北石家庄)\2024-09-23after\深入工作\文档\基于大模型的运动心理学应用研究\资料\data\question\测试题zh-100.json"  # 输入的 JSON 文件路径
    output_json_file = r"D:\xiaojun\work\knowledgeDiscovery\心理(河北石家庄)\2024-09-23after\深入工作\文档\基于大模型的运动心理学应用研究\资料\data\answer\RAG_test\Grok-3_rag_temp.json"  # 输出的 JSON 文件路径
    process_questions_from_json(input_json_file, output_json_file)