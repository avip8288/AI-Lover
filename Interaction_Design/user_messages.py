""" 
实现自动构建用户档案并且进行保存
实现两个大模型交互给出字典数据
一个是实体抽取LLM，一个是摘要LLM
另外一个大模型用来判断这个是否要写入档案
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import os
import logging
import yaml
import json

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')

current_path=os.path.abspath(__file__)
dirname=os.path.dirname(current_path)
root_name=os.path.dirname(dirname)

model_config_path=os.path.join(root_name,'config','model_config.yaml')
prompt_path=os.path.join(root_name,'config','prompt.yaml')
json_path=os.path.join(root_name,'user_message.json')

with open(model_config_path,'r') as file:
    config=yaml.safe_load(file)

with open(prompt_path,'r') as f:
    prompt=yaml.safe_load(f)

entity_llm=ChatOpenAI(model=config['model']['Google']['2.5_flash_05-20'],
                      api_key=config['model']['Google']['api_key'],
                      base_url=config['model']['Google']['api_base'])

summary_llm=ChatOpenAI(model=config['model']['Qwen']['model'],
                       api_key=config['model']['Qwen']['api_key'],
                       base_url=config['model']['Qwen']['api_base'])

judge_llm=ChatOpenAI(model=config['model']['OpenAI']['model'],
                     base_url=config['model']['OpenAI']['api_base'],
                     api_key=config['model']['OpenAI']['api_key'])


entity_prompt=prompt['Prompt']['entity']
summary_prompt=prompt['Prompt']['summary']
judge_prompt=prompt['Prompt']['judge_message']

class Messages:
    def __init__(self,text,session_id):
        self.text=text
        self.session_id=session_id
    def load_data(self):
        if os.path.exists(json_path):
            with open(json_path,'r',encoding='utf-8') as file:
                data=json.load(file)

        else:
            logging.info('用户档案没建立，现在开始建立')
            data=[]
            with open(json_path,'w',encoding='utf-8') as file:
                json_data=json.dump(data,file,ensure_ascii=False,indent=4)

        return data
    def process(self):
        data=self.load_data()
        entity_prompt_new=ChatPromptTemplate.from_template(entity_prompt)
        summary_prompt_new=ChatPromptTemplate.from_template(summary_prompt)
        judge_prompt_new=ChatPromptTemplate.from_template(judge_prompt)

        entity_chain=entity_prompt_new | entity_llm
        summary_chain=summary_prompt_new | summary_llm
        judge_chain=judge_prompt_new | judge_llm

        summary=summary_chain.invoke({'text':self.text}).content
        entity=entity_chain.invoke({'text':self.text,'entity':summary}).content

        left_idx=entity.index('{')
        right_idx=entity.index('}')
        json_str=entity[left_idx:right_idx+1]


        nums=json.loads(json_str)
        judge=judge_chain.invoke({'json_messages':nums}).content
        if '是' in judge:
            if isinstance(nums,dict):
                nums['user_id']=self.session_id
                logging.info('正确的输出了字典类型')
                data.append(nums)
                return data
            else:
                logging.info('输出的格式不是字典类型，有问题')
                return []
        else:
            return []

        
    def main(self):
        id_list=[]
        data=self.process()
        if data:
            for nums in data:
                user_id=nums['user_id']
                id_list.append(user_id)
        else:
            logging.info('输出为空列表')
            return '不增加信息到用户档案里'
            

        id_total=list(set(id_list))
        id_multi=[id for id in id_total if id_list.count(id)>=2]
        id_single=[id for id in id_total if id_list.count(id)==1]
        
        result={}
        final=[]
        for idx,value in enumerate(data):
            id=value['user_id']
            if id in id_multi:
                if id in result:
                    result[id].append(idx)
                else:
                    result[id]=[idx]
            if id in id_single:
                final.append(value)

        for key,value in result.items():
            process_list=[]
            process_dict={}
            for i in value:
                value_nums=data[i]
                process_list.append(value_nums)
            for j in process_list:
                process_dict.update(j)
            final.append(process_dict)

        with open(json_path,'w',encoding='utf-8') as file:
            json_data=json.dump(final,file,ensure_ascii=False,indent=4)

        return f'already save {json_path}'
    

if __name__=='__main__':
    text='你好可爱'
    messages=Messages(text,'hh')
    result=messages.main()
    print(result)
        







            
            
                


        



