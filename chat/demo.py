# 文件名: streamlit_app.py
import sys
import os
import streamlit as st
import yaml
import logging
import struct
from typing import TypedDict, Iterator

# --- 统一且健壮的路径设置 ---
# 将项目根目录添加到 sys.path，以便 streamlit run 可以找到模块
# 获取当前脚本的路径 -> /path/to/Project/AI-Lover/chat/demo.py
_current_script_path = os.path.abspath(__file__)
# 获取 chat 目录 -> /path/to/Project/AI-Lover/chat
_chat_dir = os.path.dirname(_current_script_path)
# 获取项目根目录 -> /path/to/Project/AI-Lover
PROJECT_ROOT_DIR = os.path.dirname(_chat_dir)

# 将项目根目录添加到Python解释器的模块搜索路径中
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# --- 导入自定义模块 ---
# 现在可以安全地从项目根目录开始导入
try:
    from chat.main import State, app, agent, embed_model
    from chat.voice import TTS
except ImportError as e:
    st.error(f"无法导入必要的模块: {e}. 请确保已正确安装 'requirements.txt' 中的所有依赖，并从项目根目录(AI-Lover)运行 `streamlit run chat/demo.py`。")
    st.stop()

# --- 日志配置 ---
LOGS_DIR = os.path.join(PROJECT_ROOT_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, 'sex_log.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# --- WAV Header Helper ---
def _create_wav_header(sample_rate: int, num_channels: int, sample_width_bytes: int, num_frames: int) -> bytes:
    """Helper function to create a WAV header for raw PCM data."""
    datasize = num_frames * num_channels * sample_width_bytes
    
    header = bytearray()
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', datasize + 36))  # ChunkSize
    header.extend(b'WAVE')
    header.extend(b'fmt ')
    header.extend(struct.pack('<I', 16))             # Subchunk1Size (16 for PCM)
    header.extend(struct.pack('<H', 1))              # AudioFormat (1 for PCM)
    header.extend(struct.pack('<H', num_channels))
    header.extend(struct.pack('<I', sample_rate))
    header.extend(struct.pack('<I', sample_rate * num_channels * sample_width_bytes))  # ByteRate
    header.extend(struct.pack('<H', num_channels * sample_width_bytes))               # BlockAlign
    header.extend(struct.pack('<H', sample_width_bytes * 8))                          # BitsPerSample
    header.extend(b'data')
    header.extend(struct.pack('<I', datasize))      # Subchunk2Size
    return bytes(header)

# --- Streamlit 界面 ---
st.set_page_config(page_title="你的专属伴侣❤️", page_icon="💬")
st.title("你的专属伴侣 ❤️")
st.caption("和你的虚拟伴侣开始聊天吧！输入 'exit'，'quit' 或 '退出' 结束对话。")


# --- 会话管理 ---
# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 检查 session_id 是否已设置
if 'session_id' not in st.session_state:
    session_id_input = st.text_input("请输入你的专属会话 ID (例如 'user123'):", key="session_id_input")
    if st.button("开始聊天", key="start_chat"):
        if session_id_input:
            st.session_state.session_id = session_id_input
            st.rerun()
        else:
            st.warning("会话 ID 不能为空。")
    st.stop() # 如果 session_id 未设置，则显示输入框并停止执行后续代码


# --- 主应用逻辑（只有在 session_id 设置后才会运行）---

# 从 session_state 获取当前 session_id
current_session_id = str(st.session_state.session_id)

# 显示聊天记录
for msg_data in st.session_state.messages:
    with st.chat_message(msg_data["role"]):
        st.markdown(msg_data["content"])

# 用户输入处理
if prompt := st.chat_input("你想对我说什么？", key="chat_input_main"):
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
        # 可以在这里加一个按钮让用户可以清除 session 并重新开始
        if st.button("结束当前会话"):
            # 清理 session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.stop()

    # 调用后端的逻辑
    with st.spinner("正在思考..."):
        try:
            logger.info(f"[Streamlit][Session: {current_session_id}] Calling agent.search_memory for long-term memory with query: \"{prompt[:50]}...\"")
            # 使用 agent.search_memory 获取长期记忆, 统一 k=3 与 main.py 逻辑对齐
            retrieved_docs = agent.search_memory(prompt, embed_model, current_session_id, k=3)

            long_term_memory_str = ""
            if isinstance(retrieved_docs, list):
                # 将文档内容拼接成字符串
                long_term_memory_str = ". ".join([doc.page_content for doc in retrieved_docs if hasattr(doc, 'page_content')])
                logger.info(f"[Streamlit][Session: {current_session_id}] Retrieved {len(retrieved_docs)} documents for long-term memory.")
            elif isinstance(retrieved_docs, str): # 兼容直接返回字符串的情况
                long_term_memory_str = retrieved_docs
                logger.info(f"[Streamlit][Session: {current_session_id}] Retrieved string from agent.search_memory.")
            else:
                logger.warning(f"[Streamlit][Session: {current_session_id}] agent.search_memory returned unexpected type: {type(retrieved_docs)}. Using empty long-term memory.")

            # 准备 LangGraph 的状态
            message_state = State(
                input=prompt,
                long_term=long_term_memory_str,
                session_id=current_session_id,
                answer=""
            )

            # 使用同步 invoke 调用 graph
            result = app.invoke(message_state, config={"configurable": {"session_id": current_session_id}})
            assistant_response = result.get('answer', '嗯...我好像不知道该说什么了。')
            logger.info(f"[Streamlit][Session: {current_session_id}] Model Response: {assistant_response}")

            # 更新并显示聊天记录
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            # --- 调用 TTS 生成并播放音频 ---
            if assistant_response:
                logger.info(f"[Streamlit][Session: {current_session_id}] Generating audio via TTS for: \"{assistant_response[:50]}...\"")

                audio_chunk_generator: Iterator[bytes] = TTS(assistant_response)

                raw_audio_data = bytearray()
                for chunk in audio_chunk_generator:
                    if chunk:
                        raw_audio_data.extend(chunk)

                logger.info(f"[Streamlit][Session: {current_session_id}] Collected {len(raw_audio_data)} bytes of raw audio data.")

                if raw_audio_data:
                    # 定义音频参数
                    sample_rate = 44100
                    num_channels = 1 # 单声道
                    sample_width_bytes = 2 # 16-bit PCM

                    num_frames = len(raw_audio_data) // (num_channels * sample_width_bytes)

                    if num_frames > 0:
                        # 创建 WAV 头部并与音频数据拼接
                        wav_header = _create_wav_header(sample_rate, num_channels, sample_width_bytes, num_frames)
                        wav_bytes = wav_header + raw_audio_data

                        # 在 Streamlit 中播放音频
                        st.audio(wav_bytes, format='audio/wav')
                        logger.info(f"[Streamlit][Session: {current_session_id}] WAV constructed and passed to st.audio.")
                    else:
                        logger.warning(f"[Streamlit][Session: {current_session_id}] Not enough raw data to form a complete audio frame.")
                else:
                    logger.warning(f"[Streamlit][Session: {current_session_id}] TTS generator yielded no data.")

        except Exception as e:
            error_message = f"处理消息时发生错误: {e}"
            st.error(error_message)
            logger.error(f"[Streamlit][Session: {current_session_id}] Error processing message: {e}", exc_info=True)
            st.session_state.messages.append({"role": "assistant", "content": f"抱歉，处理时出错了：{e}"})

    st.rerun() # 确保每次交互后刷新界面状态
