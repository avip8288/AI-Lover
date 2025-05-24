from tavily import TavilyClient
import os
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_openai import ChatOpenAI
import requests
from typing import Any, Dict, Iterator, List, Mapping, Optional,TypedDict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
"""
这个实现是否调用搜索功能
1:用LLM来判断用户语料是否调用搜索
2:用LLM把用户的语料调整为更适合搜索的语句
3:输出的搜索结果用LLM来结合输出给用户
"""

current_path=os.path.abspath(__file__)
dirname=os.path.dirname(current_path)
root_name=os.path.dirname(dirname)

config_path=os.path.join(root_name,'config','model_config.yaml')
prompt_path=os.path.join(root_name,'config','prompt.yaml')

with open(config_path,'r') as file:
    config=yaml.safe_load(file)

with open(prompt_path,'r') as f:
    prompt=yaml.safe_load(f)

judge_prompt=prompt['Prompt']['judge']
correction_prompt=prompt['Prompt']['correction']
system_prompt=prompt['Prompt']['use']
tavily_client = TavilyClient(api_key=config['model']['Tavily']['api_key'])


llm=ChatOpenAI(model=config['model']['Doubao']['model'],api_key=config['model']['Doubao']['api_key'],
               base_url=config['model']['Doubao']['api_base'])


class Process:
    def __init__(self,llm,text):
        self.llm=llm
        self.text=text

    def judge(self):
        prompt=ChatPromptTemplate.from_messages([
            ('system',judge_prompt),
            ('human','用户的语料:{text}')
        ])
        chain=prompt | self.llm
        response=chain.invoke({'text':self.text})
        return response.content

    
    def correction(self):
        response=self.judge()
        if '是' in response:
            prompt=ChatPromptTemplate.from_messages([
                ('system',correction_prompt),
                ('human','用户的语料：{text}')
            ])
            chain=prompt | self.llm
            query=chain.invoke({'text':self.text})
            return query.content
        else:
            return ''
        
    def search(self):
        query=self.correction()
        if query:
            search_result=tavily_client.search(query,max_results=3)
            search_content=[data['content'] for data in search_result['results']]
            search_final='/'.join(search_content)
            return search_final
        else:
            return ''
        
if __name__=='__main__':
    text='草泥马'
    search=Process(llm,text)
    result=search.search()
    print(result)
            




