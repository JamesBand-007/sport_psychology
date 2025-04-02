import json
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import warnings
from threading import Thread

# 忽略 Flash Attention 警告
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

class KnowledgeBase:
    def __init__(self, docs_path):
        """
        初始化知识库，从指定目录加载文档
        :param docs_path:存放知识文档的本地目录
        """
        self.docs = self.load_documents(docs_path)

    def load_documents(self, docs_path):
        """
        遍历目录，加载所有文件
        :param docs_path: 本地目录路径
        :return: 文档列表
        """
        docs = []
        for filename in os.listdir(docs_path):
            file_path = os.path.join(docs_path, filename)
            # if os.path.isfile(file_path) and filename.endswith(".txt"):
            #     with open(file_path, "r", encoding="utf-8") as f:
            #         content = f.read().strip()
            #         if content:
            #             docs.append(content)

            # if os.path.isfile(file_path) and filename.endswith(".jsonl"):
            #     with open(file_path, 'r', encoding="utf-8") as f:
            #         for line in f:
            #             data = json.loads(line)
            #             conversation = data['conversation']
            #             for item in conversation:
            #                 human = item['human']
            #                 assistant = item['assistant']
            #                 content = human + assistant
            #                 docs.append(content)


            if os.path.isfile(file_path) and filename.endswith(".json"):
                with open(file_path, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                    # print(data)
                    for item in data:
                        # print(item)
                        content = item['content']
                        docs.append(content)

        return docs



###############################################
# 2. 知识库向量化存储模块：构建并持久化向量数据库（FAISS）
###############################################
class VectorStore:
    def __init__(self, embedding_model:SentenceTransformer, index_file="faiss.index", docs_file="docs.pkl"):
        """
        初始化向量存储模块
        :param embedding_model: 本地加载的嵌入模型
        :param index_file: FAISS 索引保存的文件路径
        :param docs_file: 文档映射保存的文件路径
        """
        self.embedding_model = embedding_model
        self.dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index_file = index_file
        self.docs_file = docs_file

        # 如果存在持久化文件，则加载，否则新建
        if os.path.exists(self.index_file) and os.path.exists(self.docs_file):
            self.load_index()
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.documents = []

    def add_documents(self, docs):
        """
        对文档进行向量化并添加到FAISS索引中
        :param docs: 文档列表
        """
        embeddings = self.embedding_model.encode(docs, convert_to_numpy = True)
        # 确保数据类型为float32
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.documents.extend(docs)
    def search(self, query, top_k = 5):
        """
        根据查询文本检索相关文档
        :param query: 用户查询
        :param top_k: 检索返回的文档数量
        :return: 检索到的文档列表
        """
        query_emb = self.embedding_model.encode(query, convert_to_numpy = True)
        query_emb = np.array([query_emb]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], distance))
        return results
    def save_index(self):
        """
        将FAISS索引及文档映射持久化到文件
        """
        faiss.write_index(self.index, self.index_file)
        with open(self.docs_file, "wb") as f:
            pickle.dump(self.documents, f)
        print(f"向量数据库已保存到 {self.index_file} 和 {self.docs_file}")
    def load_index(self):
        """
        从文件加载持久化的FAISS索引及文档映射
        """
        self.index = faiss.read_index(self.index_file)
        with open(self.docs_file, "rb") as f:
            self.documents = pickle.load(f)
        print(f"向量数据库从 {self.index_file} 和 {self.docs_file} 加载成功")



###############################################
# 3. 检索增强生成对话模块：调用大模型生成回答
###############################################
class Chat:
    def __init__(self, model_path, tokenizer_path = None):
        """
        初始化生成模型模块
        :param model_path: 大模型本地路径（例如 deepseek-distill-qwen-1.5B）
        :param tokenizer_path: 分词器路径，若与模型相同可不填
        """

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, local_files_only = True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only = True)

        # 确保 pad_token_id 和 eos_token_id 正确设置
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id




    def generate(self, prompt, max_length = 1024):
        """
        根据提示生成回答
        :param prompt: 生成提示（包含检索到的知识和用户问题）
        :param max_length: 生成回答的最大长度
        :return: 生成的文本回答
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length, do_sample = True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response




###############################################
# 主程序：整合各模块，实现本地RAG系统
###############################################
def main():
    # 1. 加载知识库（确保 data/knowledge_docs 目录下存在文本文件）
    kb = KnowledgeBase(docs_path="../data/knowledge_docs")
    # print(kb.docs)
    if not kb.docs:
        print("未加载到任何文档，请检查 knowledge_docs 目录。")
        return

    # 2. 加载本地嵌入模型（请替换为你的本地模型路径）
    embedding_model = SentenceTransformer(r"D:\xiaojun\work\nlp\RAG_code\models\embedding_model\BAAI\bge-large-zh")



    # 3. 构建向量数据库：若已有持久化文件则加载，否则新建并保存
    index_file = "../data/knowledge/sport_psychology_1.index"
    docs_file = "../data/knowledge/sport_psychology_1.pkl"
    vector_store = VectorStore(embedding_model, index_file=index_file, docs_file=docs_file)
    if not os.path.exists(index_file) or not os.path.exists(docs_file):
        print("正在构建向量数据库...")
        vector_store.add_documents(kb.docs)
        vector_store.save_index()

    # 4. 加载本地大模型 deepseek-distill-qwen-1.5B（请替换为你的本地模型路径）
    chat = Chat(model_path=r"D:\xiaojun\work\nlp\RAG_code\models\LLMs\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B")

    # 5. 开启对话交互循环
    print("本地 RAG 系统已启动。输入exit退出。")
    while True:
        user_query = input("user: ").strip()
        if user_query.lower() == "exit":
            break

        # 5.1 检索相关文档（列如取 top 3）
        vector_store = VectorStore(embedding_model, index_file=index_file, docs_file=docs_file)

        retrieved_docs = vector_store.search(user_query, top_k=10)
        #  print(retrieved_docs)

        # 5.2构造生成提示：将检索到的知识与用户查询拼接
        prompt = "请结合以下知识回答问题：\n"
        for idx, doc in enumerate(retrieved_docs, start=1):
            prompt += f"[知识{idx}]: {doc[0]}\n"
        prompt += f"\n问题： {user_query}\n回答："

        # 5.3 调用大模型生成回答
        response = chat.generate(prompt)
        print("ai:",response)
        print("-" * 50)



if __name__ == '__main__':
    main()