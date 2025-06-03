"""
储存长期对话历史
"""
import json
import os
import logging
from typing import List,Dict

logging.basicConfig(level=logging.INFO,format='%(name)s-%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger('chat_history')

current_path=os.path.abspath(__file__)
dirname=os.path.dirname(current_path)
root_name=os.path.dirname(dirname)
history_path=os.path.join(root_name,'chat_history.json')


def chat_history(text,response,user_id) ->List[Dict[str,list]]:
    messages=[]
    total={}
    user={'role':'user','content':text}
    AI={'role':'assistant','content':response}
    if os.path.exists(history_path):
        with open(history_path,'r',encoding='utf-8') as file:
            data=json.load(file)
        
        if user_id in data[-1].keys():
            data[-1][user_id].append(user)
            data[-1][user_id].append(AI)
            
        else:
            logger.info('没有对应的user_id')
            messages.append(user)
            messages.append(AI)
            total[user_id]=messages
            data[0].update(total)
            
        if len(data[0][user_id])>2000:
            data[0][user_id]=data[-1][user_id][-2000:]
        
        with open(history_path,'w',encoding='utf-8') as file:
            finish=json.dump(data,file,ensure_ascii=False,indent=4)
    else:
        logger.info('没有建立chat_history文件')
        messages.append(user)
        messages.append(AI)
        total[user_id]=messages
        final=[]
        final.append(total)
        with open(history_path,'w',encoding='utf-8') as file:
            create_data=json.dump(final,file,ensure_ascii=False,indent=4)

        logger.info(f'already save {history_path}')




        
                



