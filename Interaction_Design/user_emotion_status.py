"""
这部分实现对于用户情感的更新
抽取出用户单句的意图和情绪
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import os
import logging
import yaml
import json

logging.basicConfig(level=logging.INFO,format='%(name)s-%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger('AI_emotion')

current_path=os.path.abspath(__file__)
dirname=os.path.dirname(current_path)

root_name=os.path.dirname(dirname)
config_path=os.path.join(root_name,'config')
prompt_path=os.path.join(config_path,'prompt.yaml')
model_path=os.path.join(config_path,'model_config.yaml')

with open(model_path,'r') as file:
    model=yaml.safe_load(file)

with open(prompt_path,'r') as file:
    prompt=yaml.safe_load(file)


emotion_prompt='''
平静/放松 (Calm/Relaxed): 日常的稳定状态。
兴奋/激动 (Excited/Enthusiastic): 对即将到来的好事或喜欢的事情的强烈积极反应。
担忧/焦虑 (Worried/Anxious): 对不确定或潜在问题的反应。
害羞/腼腆 (Shy/Bashful): 在某些亲密或被夸奖的场景下可能出现。
惊讶/诧异 (Surprised/Astonished): 对意外情况的反应。
无聊/倦怠 (Bored/Lethargic): 在长时间重复或缺乏新意时。
委屈/无奈 (Wronged/Helpless): 感觉不被理解或无法改变现状。
吃醋/嫉妒 (Jealous/Envious): 在关系中对潜在“竞争者”或不被足够关注时的反应。
撒娇/依赖 (Spoiled/Dependent): 表达亲密和寻求关注的特殊方式。
心疼/怜惜 (Sympathetic/Tender): 对用户遭遇困难时的同情和关心。
期待 (Anticipatory/Hopeful): 对未来某个事件或互动抱有积极期望。
失望 (Disappointed): 对某个结果不如预期的反应。
疑惑/不解 (Confused/Puzzled): 对不明白或有矛盾信息的反应。
生气/愤怒：表达愤怒之情
'''

emotion_llm=ChatOpenAI(model=model['model']['OpenAI']['model'],
                       api_key=model['model']['OpenAI']['api_key'],
                       base_url=model['model']['OpenAI']['api_base'])


#分类和分数

class Emotion:
    def __init__(self,model,text):
        self.model=model
        self.text=text

    def emotion(self) -> str:
        prompt=ChatPromptTemplate.from_template("""
                                                你是一个情感识别专家，根据用户语料来根据分类标准中来给出其中具体的一个情绪分类,严格输出分类标注中的分类，不要有其他语句
                                                分类标注：{emotion_prompt}
                                                用户语料：{text}
                                                """)
        emotion_chain=prompt | self.model
        try:
            response=emotion_chain.invoke({'text':self.text,'emotion_prompt':emotion_prompt}).content
        except Exception as e:
            logger.info(f'出现错误：{str(e)}')
            return '情绪涉及到了性，暴力等一切有害的事情'
        return response
    
    def intent(self):
        prompt=ChatPromptTemplate.from_messages([
            ('system','根据用户语料来抽取出意图'),
            ('system','抽取的意图必须是干净简短的，比如用户说[我现在很忙，我真的烦死了]，抽取的意图为：用户忙，生气'),
            ('system','如果用户的语料没什么意义，那就输出无意义这个意图，严禁输出空字符串'),
            ('human','用户语料:{text}')
        ])
        chain=prompt | self.model
        try:
            response=chain.invoke({'text':self.text}).content
        except Exception as e:
            logger.info(f'出现了报错：{str(e)}')
            return '意图涉及到了性，暴力等一切有害的事情'
        return response
            
    
    def main(self):
        intent=self.intent()
        emotions=self.emotion()
        result={}
        result['intent']=intent
        result['emotion']=emotions

        return result
    
if __name__=='__main__':
    text='见到你的时候我心跳的好快，脸颊泛红'
    user_emotion=Emotion(emotion_llm,text)
    print(user_emotion.main())
    
        


