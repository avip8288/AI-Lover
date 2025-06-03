from user_emotion_status import Emotion
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import os
import logging
import yaml
import json
import torch
import torch.nn as nn
from langchain.embeddings.base import Embeddings
import requests
from typing import List
import numpy as np

logging.basicConfig(level=logging.INFO,format='%(name)s-%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger('AI_emotion_status')

current_path=os.path.abspath(__file__)
dirname=os.path.dirname(current_path)
root=os.path.dirname(dirname)

ai_emotion_path=os.path.join(root,'AI_emotion.json')
config_path=os.path.join('config','model_config.yaml')
chat_history_path=os.path.join(root,'chat_history.json')

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
with open(ai_emotion_path,'w',encoding='utf-8') as file:
    data=json.dump(emotion_dict,file,ensure_ascii=False,indent=4)


with open(config_path,'r',encoding='utf-8') as file:
    config=yaml.safe_load(file)


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

embed_model=CustomEmbedding(config['model']['OpenAI']['model'],
                            config['model']['OpenAI']['api_key'],
                            config['model']['OpenAI']['api_base'])
'''
用户情绪和意图embedding化
messages的embedding化
#AI情感状态json文件，直接把分数输入进去就行，做成一个向量
#MOE架构计算情绪分数
#更新AI情绪分数并保存该文件
'''

class Score:
    def __init__(self,model,text,user_id):
        self.model=model
        self.text=text
        self.user_id=user_id

    def user_emotion_embed(self) -> List[float]:
        user_emotion_status=Emotion(self.model,self.text)
        user_emotion_status=user_emotion_status.main()
        user_emotion_prompt=f'用户的意图:{user_emotion_status.get('intent',None)},用户的情绪:{user_emotion_status.get('emotion',None)}'
        user_emotion_embed=embed_model.embed_query(user_emotion_prompt)
        
        return user_emotion_embed
    
    def messages_process(self,messages) -> List[str]:
        result=[]
        for nums in messages:
            nums_process=f'角色:{nums['role']},内容:{nums['content']}'
            result.append(nums_process)

        return result


    def chat_history_embed(self) -> List[float]:
        n=len(embed_model.embed_query('text'))#List[float]

        if os.path.exists(chat_history_path):
            with open(chat_history_path,'r',encoding='utf-8') as file:
                chat_history=json.load(file)
            
            data=chat_history[-1]
            chat_messages=None
            for key,values in data.items():
                if key==self.user_id:
                    chat_messages=values
                break

            if not chat_messages:
                logger.info('user_id不在对话历史记录中，是一个新的user_id')
                return [0.0]*n
            
            messages_process=self.messages_process(chat_messages)
            embed_messages=[]
            for nums in messages_process:
                embed_nums=embed_model.embed_query(nums)
                embed_messages.append(embed_nums)

            if not embed_messages:
                logger.info('没有嵌入任何向量')
                return [0.0]*n
            
            n=len(embed_messages[0])
            result_final=np.array(embed_messages)

            try:
                result_end=result_final.sum(axis=0)#[n]
                return result_end.tolist()
            
            except Exception as e:
                logger.info(f'出现报错:{str(e)}')
                return [0.0]*n

        else:
            logger.info('没有历史对话文件')
            return [0.0]*n
        
        
    def ai_emotion_data(self) -> List[float]:
        with open(ai_emotion_path,'r',encoding='utf-8') as file:
            ai_emotion=json.load(file)

        ai_emotion_embed=[score for score in ai_emotion.values()]

        return ai_emotion_embed
    
    def main(self):
        user_emotion_embed=self.user_emotion_embed()
        chat_history_embed=self.chat_history_embed()
        ai_emotion_embed=self.ai_emotion_data()

        user_tensor=torch.tensor([user_emotion_embed],dtype=torch.float32)
        chat_tensor=torch.tensor([chat_history_embed],dtype=torch.float32)
        ai_tensor=torch.tensor([ai_emotion_embed],dtype=torch.float32)

        total_embed=torch.cat([user_tensor,chat_tensor,ai_tensor],dim=1)
        n=len(user_emotion_embed)
        input_size=2*n+len(ai_emotion_embed)

        hidden_size=256
        output_size=len(ai_emotion_embed)
        
        
        

        

        



            
            


        

