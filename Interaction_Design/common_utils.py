import os
import yaml
import requests
import numpy as np
import logging
from typing import List
from langchain.embeddings.base import Embeddings

logging.basicConfig(level=logging.INFO,format='%(name)s-%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger('common_utils')


current_file_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_file_dir)
config_path = os.path.join(parent_dir, 'config', 'model_config.yaml')

if not os.path.exists(config_path):
    config_path = os.path.join(current_file_dir, 'config', 'model_config.yaml')



try:
    with open(config_path,'r',encoding='utf-8') as file:
        config=yaml.safe_load(file)
except FileNotFoundError:
    logger.error(f"配置路径没找到：{config_path}. 请检查路径")
    config = {} 



#创建AI情感json文件
emotion_dict={
'平静/放松 (Calm/Relaxed)':0,
'兴奋/激动 (Excited/Enthusiastic)': 0,
'担忧/焦虑 (Worried/Anxious)':0,
'害羞/腼腆 (Shy/Bashful)': 0,
'惊讶/诧异 (Surprised/Astonished)': 0,
'无聊/倦怠 (Bored/Lethargic)': 0,
'委屈/无奈 (Wronged/Helpless)': 0,
'吃醋/嫉妒 (Jealous/Envious)': 0,
'撒娇/依赖 (Spoiled/Dependent)': 0,
'心疼/怜惜 (Sympathetic/Tender)': 0,
'期待 (Anticipatory/Hopeful)': 0,
'失望 (Disappointed)': 0,
'疑惑/不解 (Confused/Puzzled)': 0,
'生气/愤怒':0
}

#更新情绪分数
#自定义豆包embedding大模型
class CustomEmbedding(Embeddings):
    def __init__(self,model,api_key,api_base):
        self.model=model
        self.api_key=api_key
        self.api_base=api_base

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        headers={
            'Authorization':f'Bearer {self.api_key}',
            "Content-Type": "application/json"
        }
        data={
            'input':texts,
            'encoding_format':'float',
            'model':self.model
        }
        response=requests.post(f'{self.api_base}/embeddings',headers=headers,json=data)
        result=[value['embedding'] for value in  response.json()['data']]
        return result
    
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[-1]

embed_model=CustomEmbedding(config['model']['DoubaoEmbedding']['model'],
                            config['model']['DoubaoEmbedding']['api_key'],
                            config['model']['DoubaoEmbedding']['api_base'])
