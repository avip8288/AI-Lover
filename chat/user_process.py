"""
本代码实现对于用户的语料，进行embedding之后对，在向量数据库进行相似度比分，如果分数（阈值）
大于等于0.85，则认为不应该放入向量数据库；反之则放入
"""

#faiss_db.similarity_search_with_relevance_scores
#返回的类型是List[tuple[Document,float]]


from langchain.embeddings.base import Embeddings
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.vectorstores import FAISS
from typing import List

from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from pydantic import BaseModel, Field

import os
import yaml
import logging
import math

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')

current_path=os.path.abspath(__file__)
dirname=os.path.dirname(current_path)

root_name=os.path.dirname(dirname)
config_path=os.path.join(root_name,'config','model_config.yaml')

with open(config_path,'r') as file:
    config=yaml.safe_load(file)

api_key=config['model']['DoubaoEmbedding']['api_key']
api_base=config['model']['DoubaoEmbedding']['api_base']
model=config['model']['DoubaoEmbedding']['model']

class Embedding(Embeddings):
    def __init__(self,model,api_key,api_base):
        self.model=model
        self.api_key=api_key
        self.api_base=api_base

    #做归一化向量
    #用L2范数更加贴近向量检索
    def normalize(self,data):
        n=len(data)
        final=[]
        if isinstance(data[-1],float):
            data_new=[i**2 for i in data]
            sum_data=sum(data_new)
            total=math.sqrt(sum_data)
            normalize_data=[value/total for value in data]

            return normalize_data
        else:
            for idx in range(n):
                nums=data[idx]
                data_new=[i**2 for i in nums]
                sum_data=sum(data_new)
                total=math.sqrt(sum_data)
                normalize_data=[value/total for value in nums]
                final.append(normalize_data)
            
            return final

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers={
            'Authorization':f'Bearer {self.api_key}',
            "Content-Type": "application/json"
        }
        data={
            'model':self.model,
            'input':texts,
            'encoding_format':'float'
        }
        response=requests.post(f'{self.api_base}/embeddings',json=data,headers=headers)
        result=[nums['embedding'] for nums in response.json()['data']]
        result=self.normalize(result)
        return result
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[-1]
    

embed_model=Embedding(model,api_key,api_base)

#用户输入+相似分数的比较

class Similarity:
    def __init__(self,embed_model):
        self.faiss_path='/Users/yiwise/Desktop/Project/.faiss_user_memory_storage'
        self.embed_model=embed_model
    def process(self,text,user_id,k):
        try:
            params={'k':k}
            params['filter']={'user_id':user_id}
            if os.path.exists(self.faiss_path):
                faiss_db=FAISS.load_local(
                    folder_path=self.faiss_path,
                    embeddings=self.embed_model,
                    allow_dangerous_deserialization=True
                )
                result=faiss_db.similarity_search_with_relevance_scores(text,**params)#List[Tuple[Document,float]]
                final=[]
                for doc,score in result:
                    corpus=doc.page_content
                    final.append((corpus,score))
            else:
                logging.info(f'向量数据库并不存在')
        except Exception as e:
            logging.info(f'出现错误：{str(e)}')

        if final:
            simi=sorted(final,key=lambda x:x[1],reverse=True)
            return simi
        else:
            logging.info('相似度列表为空')
            return [] 

if __name__=='__main__':
    text='你知道我的哪些信息呢'
    simi=Similarity(embed_model)
    result=simi.process(text,'hh',6)
    print(result)
