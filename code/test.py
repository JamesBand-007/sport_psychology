import os
import json

file_path = "../../../knowledgeDiscovery/心理(河北石家庄)/2024-09-23after/深入工作/资料/第一版数据/sft/train.jsonl"

with open(file_path, 'r', encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)

        if 'conversation' in data:
            conversation = data['conversation']
            for item in conversation:
                human = item['human']
                assistant = item['assistant']
                print(human+ assistant)
                print('-' * 50)