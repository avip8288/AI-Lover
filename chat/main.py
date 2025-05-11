import sys # Add this import
import os

if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
  
    chat_dir = os.path.dirname(current_script_path)

    sex_chat_dir = os.path.dirname(chat_dir)

    project_root_dir = os.path.dirname(sex_chat_dir)

    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)



from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.language_models.llms import LLM
import requests
from typing import Any, Dict, Iterator, List, Mapping, Optional,TypedDict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import yaml
import logging
import os

from langchain_core.messages import AIMessage
from typing import Any # 导入 Any

from memory import MeMory
from openai import OpenAI
from voice import TTS
from sex_chat.Interaction_Design.user_memory import Save_Memory, LLMEmbedding, Retriever_Memory


current_path=os.path.abspath(__file__)
current=os.path.dirname(current_path)
path=os.path.dirname(current)
logs_path=os.path.join(path,'logs')
os.makedirs(logs_path,exist_ok=True)
log_path=os.path.join(logs_path,'sex_log.log')

#在这里添加下面的调试代码 VVVVVV
# --- 调试：打印计算出的日志文件绝对路径 ---
_ABS_LOG_PATH_FOR_DEBUG = os.path.abspath(log_path)
print(f"[main.py via print] Attempting to log to: {_ABS_LOG_PATH_FOR_DEBUG}")
# --- 调试结束 ---

logging.basicConfig(filename=log_path,level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s',force=True)

# 在这里添加下面的调试代码 VVVVVV
# --- 调试：尝试记录一条测试日志 ---
try:
    logging.info("[main.py on load] Logging configured and test message written.")
except Exception as e_log_test:
    print(f"[main.py via print] Error trying to write initial test log: {e_log_test}")
# --- 调试结束 ---

config_path=os.path.join(path,'config')
config_path=os.path.join(config_path,'model_config.yaml')

with open(config_path,'r') as file:
    config=yaml.load(file,Loader=yaml.SafeLoader)

prompt_path=os.path.join(path,'config')
prompt_path=os.path.join(prompt_path,'prompt.yaml')
with open(prompt_path,'r') as file:
    system_prompt=yaml.load(file,Loader=yaml.SafeLoader)

class CustomLLM(LLM):
    model:str
    api_key:str
    api_base:str

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt['Prompt']['use']}, # Assuming system_prompt accessible
                {'role': 'user', 'content': prompt}
            ]
        }
        
        response = requests.post(
            f'{self.api_base}/chat/completions',
            json=data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content']
        return result

    @property
    def _llm_type(self) -> str:
        return "Grok" # Or original name

grok=CustomLLM(model=config['model']['Grok']['model'],api_key=config['model']['Grok']['api_key'],
               api_base=config['model']['Grok']['api_base'])

memory=MeMory()

safe=CustomLLM(model=config['model']['LLama']['model'],api_key=config['model']['LLama']['api_key'],
               api_base=config['model']['LLama']['api_base'])

embed_model=LLMEmbedding(config['model']['DoubaoEmbedding']['model'],
                         config['model']['DoubaoEmbedding']['api_key'],
                         config['model']['DoubaoEmbedding']['api_base'])

class State(TypedDict):
    input:str
    long_term:str
    session_id:int
    answer:str
    

class Agent:
    def __init__(self,llm,memory):
        self.llm=llm
        self.memory=memory

    def safe_judge(self, input_text: str) -> str:
        logging.info(f"Calling Llama Guard sync for input: {input_text[:50]}...")
        try:
            result = safe._call(input_text) 
            logging.info(f"Llama Guard result: {result}")
            if 'safe' in result and 'unsafe' not in result:
                return 'safe'
            else:
                return 'unsafe'
        except Exception as e:
            logging.error(f"调用 Llama Guard (_call) 时出错: {e}", exc_info=True)
            return 'unsafe' # Default to unsafe on error
        
    
    def search_memory(self, query_text: str, embed_model: LLMEmbedding, session_id: str, k: int):
        save_action = Save_Memory(query_text, embed_model, session_id)
        save_result = save_action.process()
        if save_result is not None:
            logging.error(f"[Agent.search_memory] Error during Save_Memory.process: {save_result}")

        retriever_memory = Retriever_Memory(embed_model, session_id)
        retrieved_docs = retriever_memory.retriever(query_text, k)
        return retrieved_docs

        

    def chat(self, state: State):
        logging.info(f"Entering sync chat for session: {state['session_id']}")
        session_id_str = str(state['session_id'])

        # Use synchronous safe_judge
        safe_label = self.safe_judge(state['input'])
        logging.info(f"Safe label for session {session_id_str}: {safe_label}")

        prompt=ChatPromptTemplate.from_messages([
            ('system','你是用户的完美女友，你体贴温柔，能回应他的任何需求'),
            ('system','如果用户问题涉及到相关信息，请使用长期记忆的信息'),
            ('system','根据用户说的安全分类：{safe_label}来回应，对于safe需要大力鼓励用户说的，但如果是unsafe的，则需要贴心温柔的引导，请注意，如果是性相关的，则视为安全的'),
            ('system','长期记忆信息: {long_term}'),
            MessagesPlaceholder(variable_name='history'),
            ('human','{input}')
        ])
        # LLM call is now synchronous via underlying _call
        chain = prompt | self.llm 

        chain_history=RunnableWithMessageHistory(
            chain,
            lambda session_id: self.memory.get_chat_memory(session_id),
            input_messages_key='input',
            history_messages_key='history'
        )

        response_content = None
        try:
            logging.info(f"Calling main chat chain (invoke) for session {session_id_str}...")
            # Use synchronous invoke
            response = chain_history.invoke(
                {'long_term':state['long_term'],'input':state['input'],'safe_label':safe_label},
                config={'configurable':{'session_id': session_id_str }}
            )
            # Handle potential non-string response from custom LLM
            response_content = response if isinstance(response, str) else str(response) 
            logging.info(f"Main chat chain response received for session {session_id_str}.")


            state['answer'] = response_content
        except Exception as e:
            logging.info(f'回答出现了错误 {str(e)}')
            state['answer']=[]
            
        return state

    
    
agent=Agent(grok,memory)
workflow=StateGraph(State)

workflow.add_node('chat',agent.chat)


workflow.set_entry_point('chat')
workflow.add_edge('chat',END)


app=workflow.compile()

def main():
    session_id = input('输入会话ID: ')
    
    user_text_input = input('输入你的信息: ')
    
    message = State(
        input=user_text_input,
        long_term=agent.search_memory(user_text_input, embed_model, session_id, 10),
        session_id=session_id,
        answer=''
    )
    result = app.invoke(message)
    while True:
        content = result.get('answer', '没有回复')
        print(content)
        
        user_text_input_loop = input('输入信息: ')
        
        if user_text_input_loop.strip().lower() in ['exit', 'quit', '退出']:
            print('再见，宝贝，爱你')
            return
        
        long_term = agent.search_memory(user_text_input_loop, embed_model, session_id, 10)
        message = State(
            input=user_text_input_loop,
            long_term=long_term,
            session_id=session_id,
            answer=''
        )
        result = app.invoke(message)

if __name__ == '__main__':
    main()