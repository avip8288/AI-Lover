# 文件名: streamlit_app.py

import streamlit as st
import os
import yaml
import logging
from typing import TypedDict, Any
# Remove asyncio and nest_asyncio imports
# import asyncio
# import nest_asyncio
# nest_asyncio.apply()

# --- 导入 ---
try:
    from chat.main import State, app, memory as memory_handler
    from chat.voice import TTS
    # Removed: from chat.memory import MeMory (using memory_handler instance from chat.main)
    # Removed: from langgraph.graph import END, StateGraph (not used directly in demo.py)
except ImportError as e:
    st.error(f"无法导入必要的模块: {e}")
    st.stop()

# --- 日志配置 (保持不变) ---
logs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_path, exist_ok=True)
log_path = os.path.join(logs_path, 'sex_log.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# --- Streamlit 界面 (标题等不变) ---
st.set_page_config(page_title="你的专属女友❤️", page_icon="💬")
st.title("你的专属女友 ❤️")
st.caption("和你的虚拟女友开始聊天吧！输入 'exit'，退出或者quit 结束对话。")

# --- 会话管理 (不变) ---
if 'session_id' not in st.session_state:
    session_id_input = st.text_input("请输入你的专属会话 ID (例如 'user123'):", key="session_id_input")
    if session_id_input:
        st.session_state.session_id = session_id_input
        st.rerun()
    else:
        st.info("请输入一个会话 ID 以开始聊天。")
        st.stop()
current_session_id = str(st.session_state.session_id)
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 显示聊天记录 (不变) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 移除异步处理函数 --- 
# async def process_user_input(user_prompt: str): ...

# --- 移除直接测试按钮 --- 
# st.divider()
# st.subheader("直接网络调用测试")
# if st.button("测试直接调用 Grok API"): ...

# --- 用户输入处理 (恢复为简单的同步调用) ---
if prompt := st.chat_input("你想对我说什么？"):
    # 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    logger.info(f"[Streamlit][Session: {current_session_id}] User Input: {prompt}")

    if prompt.strip().lower() in ['exit', 'quit', '退出']:
        farewell_message = "好的，宝贝，下次再聊！❤️"
        st.session_state.messages.append({"role": "assistant", "content": farewell_message})
        with st.chat_message("assistant"):
            st.markdown(farewell_message)
        st.stop()
    else:
        # 直接调用同步的 app.invoke
        with st.spinner("正在思考..."):
            try:
                # 获取长期记忆 (假设同步)
                long_term_memory_str = memory_handler.get_long_term(current_session_id)

                message = State(
                    input=prompt,
                    long_term=long_term_memory_str,
                    session_id=current_session_id,
                    answer=""
                )

                # 使用同步 invoke
                result = app.invoke(message, config={"configurable": {"session_id": current_session_id}})

                # 处理结果
                assistant_response = result.get('answer', '嗯...我好像不知道该说什么了。')
                logger.info(f"[Streamlit][Session: {current_session_id}] Model Response: {assistant_response}")

                # 添加到消息历史
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                # 立即显示助手的回复
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)

                # --- 修改：调用 TTS 获取音频数据并在客户端播放 ---
                if assistant_response:
                    logger.info(f"[Streamlit][Session: {current_session_id}] Attempting to get audio data via TTS for: {assistant_response[:50]}...")
                    audio_bytes = TTS(assistant_response) # TTS 现在返回字节或 None

                    if audio_bytes:
                        logger.info(f"[Streamlit][Session: {current_session_id}] Received audio data, {len(audio_bytes)} bytes. Playing in browser.")
                        st.audio(audio_bytes, format='audio/wav') # <--- 使用 st.audio 播放
                    else:
                        logger.warning(f"[Streamlit][Session: {current_session_id}] TTS did not return audio data.")
                        # （可选）可以给用户一个提示，比如 st.toast("抱歉，语音暂时无法播放。")
                # --- 音频处理结束 ---

            except Exception as e:
                error_message = f"处理消息时发生错误: {e}"
                st.error(error_message)
                logger.error(f"[Streamlit][Session: {current_session_id}] Error processing message: {e}", exc_info=True)
                st.session_state.messages.append({"role": "assistant", "content": f"抱歉，处理时出错了：{e}"})

        # Streamlit 会在脚本结束时自动刷新，通常不需要手动 rerun
        # st.rerun()

# --- (可选) 清除聊天记录按钮 (保持不变) ---
# ...
