'''
生成MOE训练数据
因为是自动生成，所以数据质量不能很好保证
如果想要高质量的数据，请通过人工方式来校对

一
创造对话历史
二
制造label
三
返回dataloader形式
'''

import torch
import torch.nn as nn
import os
import logging
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from .common_utils import embed_model, emotion_dict
from .user_emotion_status import Emotion
import numpy as np

logging.basicConfig(level=logging.INFO,format='%(name)s-%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger('MOE_Data')

current_path=os.path.abspath(__file__)
dirname=os.path.dirname(current_path)
root=os.path.dirname(dirname)
prompt_path=os.path.join(root,'config','prompt.yaml')
config_path=os.path.join(root,'config','model_config.yaml')
with open(config_path,'r') as file:
    config=yaml.safe_load(file)

with open(prompt_path,'r') as f:
    prompt=yaml.safe_load(f)

logger.info(f'成功加载config文件:{config_path}')
AI=ChatOpenAI(model=config['model']['Google']['2.5_flash_05-20'],
              api_key=config['model']['Google']['api_key'],
              base_url=config['model']['Google']['api_base']
)
User=ChatOpenAI(model=config['model']['OpenAI']['model'],
                api_key=config['model']['OpenAI']['api_key'],
                base_url=config['model']['OpenAI']['api_base'])

Check_LLM=ChatOpenAI(model=config['model']['Grok']['model'],
                     api_key=config['model']['Grok']['api_key'],
                     base_url=config['model']['Grok']['api_base'])

user_prompt=prompt['Prompt']['User_Create']
AI_prompt=prompt['Prompt']['use']

class History:
    def create_ddata(self):
        count=50
        messages=[]
        messages.append({'role':'user','content':'你好啊'})
        for i in range(count):
            user_messages=[
                SystemMessage(content=user_prompt)
                
            ]
            ai_messages=[
                SystemMessage(content=AI_prompt),
                HumanMessage(content='你好啊')
            ]
            ai_content=AI.invoke(ai_messages).content
            messages.append({'role':'assistant','content':ai_content})
            human_message=HumanMessage(content=ai_content)
            user_messages.append(human_message)

            user_content=User.invoke(user_messages).content
            messages.append({'role':'user','content':user_content})
            user_message=HumanMessage(content=user_content)
            ai_messages.append(user_message)

        return messages
    
    def main(self):
        total=10
        result=[]
        for i in tqdm(range(total)):
            messages=self.create_ddata()
            result.append(messages)
        
        with open(os.path.join(root,'chat_train.json'),'w',encoding='utf-8') as file:
            json.dump(result,file,ensure_ascii=False,indent=4)

class Label:
    EMOTION_LIST = [
    '平静/放松', '兴奋/激动', '担忧/焦虑', '害羞/腼腆', '惊讶/诧异', 
    '无聊/倦怠', '委屈/无奈', '吃醋/嫉妒', '撒娇/依赖', '心疼/怜惜', 
    '期待', '失望', '疑惑/不解', '生气/愤怒'
]
    def main(self,dialog):
        emotion2idx={emotion:idx for idx,emotion in enumerate(self.EMOTION_LIST)}
        idx2emotion={idx:emotion for idx,emotion in enumerate(self.EMOTION_LIST)}
        
        class EmotionScores(BaseModel):
            calm: float = Field(description="平静/放松的程度, 0.0到1.0")
            excited: float = Field(description="兴奋/激动的程度, 0.0到1.0")
            worried: float = Field(description="担忧/焦虑的程度, 0.0到1.0")
            shy: float = Field(description="害羞/腼腆的程度, 0.0到1.0")
            surprised: float = Field(description="惊讶/诧异的程度, 0.0到1.0")
            bored: float = Field(description="无聊/倦怠的程度, 0.0到1.0")
            wronged: float = Field(description="委屈/无奈的程度, 0.0到1.0")
            jealous: float = Field(description="吃醋/嫉妒的程度, 0.0到1.0")
            spoiled: float = Field(description="撒娇/依赖的程度, 0.0到1.0")
            sympathetic: float = Field(description="心疼/怜惜的程度, 0.0到1.0")
            anticipatory: float = Field(description="期待的程度, 0.0到1.0")
            disappointed: float = Field(description="失望的程度, 0.0到1.0")
            confused: float = Field(description="疑惑/不解的程度, 0.0到1.0")
            angry: float = Field(description="生气/愤怒的程度, 0.0到1.0")

        structured_llm=Check_LLM.with_structured_output(EmotionScores)

        prompt=ChatPromptTemplate.from_template("""
你是一个专业的情感分析师。请仔细分析对话记录，并根据我们定义的14种情绪，为每一种情绪打分。
分数范围从0.0到1.0。0.0表示完全不包含该情绪，1.0表示该情绪表现得极为强烈。
请严格按照我们定义的格式输出你的分析结果。

14种情绪:{emotion}
对话记录:{dialog}                                                                  
"""

        )
        chain=prompt | structured_llm
        scores_object=chain.invoke({'emotion':self.EMOTION_LIST,'dialog':dialog})
        y_label_vector = [
    scores_object.calm, scores_object.excited, scores_object.worried, 
    scores_object.shy, scores_object.surprised, scores_object.bored,
    scores_object.wronged, scores_object.jealous, scores_object.spoiled,
    scores_object.sympathetic, scores_object.anticipatory, scores_object.disappointed,
    scores_object.confused, scores_object.angry
]
        return y_label_vector
    

class Create_Dataset(Dataset):
    def __init__(self,history_path:str,label_generator:Label):
        self.x_data=[]
        self.y_data=[]

        with open(history_path,'r',encoding='utf-8') as file:
            chat_history=json.load(file)

        logger.info(f'成功加载对话历史数据{len(chat_history)}条')

        user_emotion_data=Emotion(User,'见到你很开心').main()
        user_str=f"用户意图:{user_emotion_data.get('intent',None)},用户情绪:{user_emotion_data.get('emotion',None)}"
        user_embed=embed_model.embed_query(user_str)
        user_tensor=torch.tensor([user_embed],dtype=torch.float32)

        ai_emotion_data=[value for value in emotion_dict.values()]
        ai_tensor=torch.tensor([ai_emotion_data],dtype=torch.float32)

        for conversation in tqdm(chat_history):
            dialog_str=json.dumps(conversation)
            label_embed=label_generator.main(dialog_str)
            self.y_data.append(torch.tensor(label_embed,dtype=torch.float32))

            chat_str=[f"角色:{data['role']},内容:{data['content']}" for data in conversation]
            embed_messages=embed_model.embed_documents(chat_str)
            if not embed_messages:
                chat_embed=[0.0]*len(user_embed)
            else:
                chat_embed=np.array(embed_messages).mean(axis=0).tolist()
            chat_tensor=torch.tensor([chat_embed],dtype=torch.float32)

            total_embed=torch.cat([user_tensor,ai_tensor,chat_tensor],dim=1)
            self.x_data.append(total_embed.squeeze(0))

    def __len__(self):
        return len(self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
def main():
    history_path=os.path.join(root,'chat_train.json')
    label=Label()
    moe_dataset=Create_Dataset(history_path,label)
    train_loader=DataLoader(moe_dataset,batch_size=16,shuffle=True)

    return train_loader


if __name__=='__main__':
    history=History()
    history.main()
    

        
            
            


