from .user_emotion_status import Emotion,emotion_llm
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
from typing import List
import copy
import numpy as np


from .MOE_score import MOEScore,MOETrain
from .MOE_data import main,User
from .common_utils import embed_model, emotion_dict

logging.basicConfig(level=logging.INFO,format='%(name)s-%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger('AI_emotion_status')

current_path=os.path.abspath(__file__)
dirname=os.path.dirname(current_path)
root=os.path.dirname(dirname)

ai_emotion_path=os.path.join(root,'AI_emotion.json')
chat_train_path=os.path.join(root,'chat_train.json')#用于训练
chat_history_path=os.path.join(root,'chat_history.json')#用于推理


'''
用户情绪和意图embedding化
messages的embedding化
#AI情感状态json文件，直接把分数输入进去就行，做成一个向量
#MOE架构计算情绪分数
#更新AI情绪分数并保存该文件
'''

class MOEModule:
    def __init__(self,model):
        self.model=model

    def user_emotion_embed(self) -> List[float]:
        user_emotion_status=Emotion(self.model,'见到你我很开心')
        user_emotion_status=user_emotion_status.main()
        user_emotion_prompt=f"用户的意图:{user_emotion_status.get('intent',None)},用户的情绪:{user_emotion_status.get('emotion',None)}"
        user_emotion_embed=embed_model.embed_query(user_emotion_prompt)
        
        return user_emotion_embed
    
    def messages_process(self,messages) -> List[str]:
        result=[]
        for nums in messages:
            nums_process=f"角色:{nums['role']},内容:{nums['content']}"
            result.append(nums_process)

        return result

    #chat_history:List[List[Dict[str,str]]]
    def chat_history_embed(self) -> List[float]:
        n=len(embed_model.embed_query('text'))#List[float]

        if os.path.exists(chat_train_path):
            with open(chat_train_path,'r',encoding='utf-8') as file:
                chat_history=json.load(file)
            
            chat_messages=chat_history[-1]
           
            
            
            if not chat_messages:
                logger.info('训练数据集的为空')
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
                result_end=result_final.mean(axis=0)#[n]
                return result_end.tolist()
            
            except Exception as e:
                logger.info(f'出现报错:{str(e)}')
                return [0.0]*n

        else:
            logger.info('没有历史对话文件')
            return [0.0]*n
        
        
    def ai_emotion_data(self):
        try:
            with open(ai_emotion_path,'r',encoding='utf-8') as file:
                data=json.load(file)

        except Exception as e:
            with open(ai_emotion_path,'w',encoding='utf-8') as file:
                nums=json.dump(emotion_dict,file,ensure_ascii=False,indent=4)
                data=emotion_dict.copy()
        finally:
            emotion_embed=[value for value in data.values()]
            return emotion_embed

    
    def main(self):
        '''
        List[float]
        '''
        user_emotion_embed=self.user_emotion_embed()
        chat_history_embed=self.chat_history_embed()
        ai_emotion_embed=self.ai_emotion_data()

        user_tensor=torch.tensor([user_emotion_embed],dtype=torch.float32)
        chat_tensor=torch.tensor([chat_history_embed],dtype=torch.float32)
        ai_tensor=torch.tensor([ai_emotion_embed],dtype=torch.float32)

        total_embed=torch.cat([user_tensor,ai_tensor,chat_tensor],dim=1)
        
        input_size=total_embed.size(1)#输入特征

        hidden_size=256
        output_size=len(ai_emotion_embed)

        return input_size,hidden_size,output_size,total_embed
    
    def train_and_save_model(self):
        input_size,hidden_size,output_size,_=self.main()
        #这里是mac的写法
        #如果用的cuda,请用device='cuda:0' if torch.cuda.is_available() else 'cpu'
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f'正在使用的device:{device}')

        moe_score=MOEScore(input_size,hidden_size,output_size)
        moe_score.to(device)
        moe_train=MOETrain(moe_score)
        train_loader=main()
        moe_train.train(train_loader)
        model_path=os.path.join(root,'trained_moe_model.pth')
        torch.save(moe_score.state_dict(), model_path)

        
        
        
"""
这部分加载训练好的MOE模型
然后输出成一个AI情感状态的json文件
"""
class Score:
    hidden_size=256#注意这里的类变量的超参数的值要和训练时的超参数的值一致
    output_size=14#情绪类别
    model_path=os.path.join(root,'trained_moe_model.pth')
    chat_history=os.path.join(root,'chat_history.json')

    emotion_list = [
        "平静/放松 (Calm/Relaxed)",
        "兴奋/激动 (Excited/Enthusiastic)",
        "担忧/焦虑 (Worried/Anxious)",
        "害羞/腼腆 (Shy/Bashful)",
        "惊讶/诧异 (Surprised/Astonished)",
        "无聊/倦怠 (Bored/Lethargic)",
        "委屈/无奈 (Wronged/Helpless)",
        "吃醋/嫉妒 (Jealous/Envious)",
        "撒娇/依赖 (Spoiled/Dependent)",
        "心疼/怜惜 (Sympathetic/Tender)",
        "期待 (Anticipatory/Hopeful)",
        "失望 (Disappointed)",
        "疑惑/不解 (Confused/Puzzled)",
        "生气/愤怒"
    ]

    def __init__(self):
        self.device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model=self._load_model()

    def _load_model(self):
        moe_model=MOEModule(User)
        input_size,_,_,_=moe_model.main()
        
        inference_model=MOEScore(
            input_size,
            self.hidden_size,
            self.output_size
        )
        inference_model.load_state_dict(torch.load(self.model_path,map_location=self.device))
        inference_model.to(self.device)
        return inference_model
    
    def load_data(self):

        if os.path.exists(self.chat_history):
            with open(self.chat_history,'r',encoding='utf-8') as file:
                chat_history=json.load(file)

        else:
            logger.info(f'确保该文件路径存在: {self.chat_history}')
            raise FileNotFoundError('对话历史文件不存在，请确保该文件存在')
            
        
        if os.path.exists(ai_emotion_path):
            with open(ai_emotion_path,'r',encoding='utf-8') as f:
                ai_data=json.load(f)

        else:
            logger.info('ai情感状态路径文件不存在')
            ai_data=emotion_dict

        return chat_history,ai_data

            

    
    def predict(self,user_id):
        self.model.eval()

        chat_history,ai_data=self.load_data()
        if chat_history:
            history_data=chat_history[-1]
            messages=history_data.get(user_id,None)

            if messages:
                if messages[-2].get('role') == 'user':
                    user_data=messages[-1].get('content')
                    user_status=Emotion(emotion_llm,user_data).main()
                    user_process=f"用户的意图:{user_status.get('intent')},用户的情绪：{user_status.get('emotion')}"
                    user_embed=embed_model.embed_query(user_process)
                    user_tensor=torch.tensor([user_embed],dtype=torch.float32)
                else:
                    logger.info('聊天历史倒数第二条信息需要是AI') 
                    raise ValueError('倒数第二条角色需要是assistant')
                     

                messages_as_list=[f"角色：{nums.get('role')},内容：{nums.get('content')}" for nums in messages]
                messages_embed=embed_model.embed_documents(messages_as_list)
                chat_history_embedding=np.array(messages_embed).mean(axis=0).tolist()#List[float]
                messages_tensor=torch.tensor([chat_history_embedding],dtype=torch.float32)

            else:
                logger.info('没有该用户的聊天历史')
                raise FileNotFoundError('请一定要有先有对应用户的对话历史文件')
                
        else:
            logger.info('对话历史为空')
            raise ValueError('请一定要有先有对话历史文件')

        if ai_data:
            ai_embed=[value for value in ai_data.values()]
            ai_tensor=torch.tensor([ai_embed],dtype=torch.float32)

        else:
            ai_data=emotion_dict
            ai_embed=[value for value in ai_data.values()]
            ai_tensor=torch.tensor([ai_embed],dtype=torch.float32)
            logger.info('ai情感状态文件为空,使用默认配置文件')

        input_tensor=torch.cat([user_tensor,ai_tensor,messages_tensor],dim=1)
        input_tensor_device=input_tensor.to(self.device)
        with torch.no_grad():
            final_output,_=self.model(input_tensor_device)

        emotion_score=torch.sigmoid(final_output).squeeze().cpu().numpy().tolist()

        return emotion_score

    def main(self,user_id):
        idx2emotion={idx:emotion for idx,emotion in enumerate(self.emotion_list)}
        emotion_score=self.predict(user_id)
        AI_emotion_status_data={}
        n=len(emotion_score)
        for i in range(n):
            emotion=idx2emotion[i]
            score=emotion_score[i]
            AI_emotion_status_data[emotion]=score

        _,ai_data=self.load_data()
        AI_emotion_final={}
        
        emotion_name=list(ai_data.keys())
        for name in emotion_name:
            AI_emotion_final[name]=AI_emotion_status_data[name]+ai_data[name]

        with open(ai_emotion_path,'w',encoding='utf-8') as file:
            json.dump(AI_emotion_final,file,ensure_ascii=False,indent=4)

 
if __name__=='__main__':
    '''
    运行MOE_data.py生成训练集
    然后运行这里的保存模型
    '''
    moe_module=MOEModule(User)
    moe_module.train_and_save_model()
    logger.info('模型训练并保存成功')