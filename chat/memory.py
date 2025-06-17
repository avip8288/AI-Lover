from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_store={}
long_term_memory={}

class MeMory:
    def __init__(self):
        pass

    def get_chat_memory(self,session_id:str):
        if session_id not in chat_store:
            chat_store[session_id]=ChatMessageHistory()

        if len(chat_store[session_id].messages)>600:
            chat_store[session_id].messages=chat_store[session_id].messages[-600:]

        return chat_store[session_id]
    '''
    def long_term(self,session_id:str,input:str):
        if session_id not in long_term_memory:
            long_term_memory[session_id]=[]
        
        input=input.replace(',','').replace('，','')
        if len(input)>10:
            long_term_memory[session_id].append(f'user said:{input}')
        
        if len(long_term_memory[session_id])>50:
            long_term_memory[session_id]=long_term_memory[session_id][-50:]

    def get_long_term(self,session_id:str):
        return '. '.join(long_term_memory.get(session_id,[]))
    '''
    
        

        
            

        

        