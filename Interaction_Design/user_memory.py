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

import yaml
import os
import logging
import math

""" 
本轮主要是通过FAISS来存储相应用户编号的记忆
通过检索对应的用户的记忆来输出
"""
current_path=os.path.abspath(__file__)
dirname_path=os.path.dirname(current_path)

mkdir_path=os.path.dirname(dirname_path)
config_path=os.path.join(mkdir_path,'config')

model_config_path=os.path.join(config_path,'model_config.yaml')

with open(model_config_path,'r') as file:
    config=yaml.safe_load(file)


#自定义embedding模型

class LLMEmbedding(Embeddings):
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
        result=[value['embedding'] for value in response.json()['data']]
        final=self.normalize(result)
        return final
    
    def embed_query(self, text: str) -> List[float]:
        result=self.embed_documents([text])[-1]
        return result
    
Embeddings_model=LLMEmbedding(config['model']['DoubaoEmbedding']['model'],config['model']['DoubaoEmbedding']['api_key'],
                              config['model']['DoubaoEmbedding']['api_base'])

# --- 更稳健地定义 FAISS_index_path --- 
# 获取 user_memory.py 文件所在的目录 (Project/sex_chat/Interaction_Design/)
_USER_MEMORY_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取 sex_chat 目录 (Project/sex_chat/)
_SEX_CHAT_DIR = os.path.dirname(_USER_MEMORY_FILE_DIR)
# 获取项目根目录 (Project/)
_PROJECT_ROOT_DIR = os.path.dirname(_SEX_CHAT_DIR)

# 定义 FAISS 索引存储在项目根目录下的一个特定文件夹中
_FAISS_INDEX_SUBDIR_NAME = ".faiss_user_memory_storage" # 使用点号开头，使其在某些系统中为隐藏目录
FAISS_index_path = os.path.join(_PROJECT_ROOT_DIR, _FAISS_INDEX_SUBDIR_NAME)

# 立即打印一次计算出的绝对路径，用于调试，确保 main.py 和 demo.py 运行时此路径一致
_ABS_FAISS_PATH_FOR_DEBUG = os.path.abspath(FAISS_index_path)
print(f"[user_memory.py on load via print] FAISS Index Path set to: {_ABS_FAISS_PATH_FOR_DEBUG}")
logging.info(f"[user_memory.py on load] FAISS Index Path set to: {_ABS_FAISS_PATH_FOR_DEBUG}")
# --- FAISS_index_path 定义结束 ---

#保存用户的记忆
"""
属于用户的单条语料，然后根据user_id来进行保存
"""
class Save_Memory:
    def __init__(self,text:str,model,user_id):
        self.text=text
        self.model=model
        self.user_id=user_id
        # 使用全局定义的、相对于文件位置的 FAISS_index_path
        self.index_path = FAISS_index_path

    def process(self):
        abs_index_path = os.path.abspath(self.index_path)
        logging.info(f"[Save_Memory] Processing memory for user_id: {self.user_id}. Target FAISS index path: {abs_index_path}")

        try:
            # 关键：在任何 FAISS 操作之前，确保目标目录存在
            os.makedirs(self.index_path, exist_ok=True)
            logging.info(f"[Save_Memory] Ensured FAISS directory exists: {abs_index_path}")
        except OSError as e:
            logging.error(f"[Save_Memory] Fatal: Could not create FAISS directory {abs_index_path}: {e}", exc_info=True)
            return f"创建向量数据库目录时出错: {str(e)}"

        # FAISS 元数据通常需要字符串、整数、浮点数或布尔值
        spliter=RecursiveCharacterTextSplitter(chunk_size=50,chunk_overlap=20)

        text_split=spliter.split_text(self.text)
        current_user_id=[{'user_id':self.user_id}]*len(text_split)
        faiss_db = None
        # 检查实际的索引文件是否存在，以此判断是否是已存在的索引
        faiss_index_file_concrete = os.path.join(self.index_path, "index.faiss")

        if os.path.exists(faiss_index_file_concrete):
            try:
                logging.info(f"[Save_Memory] Attempting to load existing FAISS DB from {abs_index_path}")
                faiss_db = FAISS.load_local(
                    folder_path=self.index_path,
                    embeddings=self.model,
                    allow_dangerous_deserialization=True
                )
                logging.info(f"[Save_Memory] Successfully loaded existing FAISS DB. Adding texts.")
                faiss_db.add_texts(texts=text_split, metadatas=current_user_id)
            except Exception as e:
                logging.error(f"[Save_Memory] Error loading or adding to existing FAISS DB from {abs_index_path}. Will attempt to re-initialize. Error: {str(e)}", exc_info=True)
                faiss_db = None # 明确设置为 None 以触发重新初始化
        else:
            logging.info(f"[Save_Memory] No existing FAISS index file found at {faiss_index_file_concrete}. Will initialize new DB.")
            # faiss_db 已经是 None 或者在上一个 except 中被设为 None

        if faiss_db is None: # 如果是首次创建或加载失败，则初始化新的数据库
            try:
                logging.info(f"[Save_Memory] Initializing new FAISS DB at {abs_index_path}")
                faiss_db = FAISS.from_texts(
                    texts=text_split,
                    embedding=self.model,
                    metadatas=current_user_id
                )
                logging.info(f"[Save_Memory] Successfully initialized new FAISS DB.")
            except Exception as e:
                logging.error(f"[Save_Memory] Error initializing new FAISS DB at {abs_index_path}: {str(e)}", exc_info=True)
                return f'初始化向量数据库出现错误：{str(e)}'
            
        if faiss_db: # 确保 faiss_db 对象存在 (无论是加载的还是新建的)
            try:
                logging.info(f"[Save_Memory] Saving FAISS DB to {abs_index_path}")
                faiss_db.save_local(self.index_path)
                logging.info(f"[Save_Memory] Successfully saved FAISS DB.")
                return None # 表示成功，无错误信息返回
            except Exception as e:
                logging.error(f"[Save_Memory] Error saving FAISS DB to {abs_index_path}: {str(e)}", exc_info=True)
                return f'数据库保存时出现错误:{str(e)}'
        else:
            logging.error(f"[Save_Memory] FAISS DB object is None after load/initialization attempt at {abs_index_path}, cannot save.")
            return "FAISS数据库对象为空，无法保存"
            

#检索向量功能实现
class Retriever_Memory:
    def __init__(self,emb_model,user_id):
        self.emb_model=emb_model
        self.user_id=user_id
        # 使用全局定义的、相对于文件位置的 FAISS_index_path
        self.index_path=FAISS_index_path

    def retriever(self,text,k):
        abs_index_path = os.path.abspath(self.index_path)
        logging.info(f"[Retriever_Memory] Attempting to retrieve from FAISS index: {abs_index_path} for user_id: {self.user_id}, query: '{text[:50]}...', k={k}")

        # 首先检查目录是否存在，如果目录本身就不存在，那肯定无法加载
        if not os.path.isdir(abs_index_path):
            logging.warning(f"[Retriever_Memory] FAISS index directory does not exist: {abs_index_path}")
            return []
        
        # 进一步检查关键的索引文件是否存在，例如 'index.faiss'
        # 如果连这个文件都没有，load_local 也会失败
        faiss_index_file_concrete = os.path.join(self.index_path, "index.faiss")
        if not os.path.exists(faiss_index_file_concrete):
            logging.warning(f"[Retriever_Memory] FAISS index file ({faiss_index_file_concrete}) does not exist. No data to load.")
            return []

        try:
            faiss_db=FAISS.load_local(
                folder_path=self.index_path,
                embeddings=self.emb_model,
                allow_dangerous_deserialization=True
            )
            logging.info(f"[Retriever_Memory] Successfully loaded FAISS index from: {abs_index_path}")

            search_params = {'k':k}
            # 只有当 user_id 确实存在时才添加到过滤器，避免空的过滤器值
            if self.user_id:
                search_params['filter'] = {'user_id': str(self.user_id)} # 确保 user_id 是字符串
            
            logging.info(f"[Retriever_Memory] Performing similarity search with params: {search_params}")
            # 修正 similarity_search 的调用方式，使用 ** 解包参数字典
            results=faiss_db.similarity_search(query=text, **search_params)
            logging.info(f"[Retriever_Memory] Search completed. Found {len(results)} results.")
            answer=[value.page_content for value in results]
            return answer
        except Exception as e:
            logging.error(f"[Retriever_Memory] Error during FAISS search or load from {abs_index_path}: {str(e)}", exc_info=True)
            return []

            

if __name__=='__main__':
    text=['你好啊，朋友']
    user_id='3'
    embed_model=Embeddings_model.embed_documents(text)
    import ipdb;ipdb.set_trace()
    
    
        
        


