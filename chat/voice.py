import requests
import yaml
# import simpleaudio as sa # simpleaudio 不再需要在服务器端播放
import io
import os

# Get the directory of the current script (voice.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one directory to get to the 'chat' parent directory (which should be 'sex_chat')
project_chat_dir = os.path.dirname(current_script_dir)
# Construct the path to the config file
config_file_path = os.path.join(project_chat_dir, 'config', 'model_config.yaml')

with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

url = config['model']['Cartesia']['url']
voice_id = config['model']['Cartesia']['voice_id']
api_key = config['model']['Cartesia']['api_key']

def TTS(text: str):
    payload = {
        "model_id": "sonic-2",
        "transcript": text, 
        "language": "zh",
        "voice": {"mode": "id", "id": voice_id}, 
        "output_format": {
            "container": "wav", 
            "encoding": "pcm_s16le",
            "sample_rate": 44100,
        },
        "speed": "slow"
    }
    headers = {
        "Cartesia-Version": "2025-04-16",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20) # 增加超时
        print(f"Status Code from Cartesia API: {response.status_code}") # 更明确的日志
        response.raise_for_status() # 如果请求失败 (如 4xx, 5xx)，会抛出异常

        if response.content:
            return response.content # <--- 直接返回音频字节数据
        else:
            print("Cartesia API returned empty audio content.") # 更明确的日志
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"Cartesia API HTTP error occurred: {http_err} - Response: {response.text if response else 'No response'}") # 记录响应内容
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Request to Cartesia API failed: {req_err}")
        return None
    except Exception as e:
        print(f"TTS function encountered an unknown error: {e}")
        return None

if __name__ == '__main__':
    sample_text = "你好，这是一个测试语音，用于验证TTS功能。"
    print(f"Testing TTS with: '{sample_text}'")
    audio_data = TTS(sample_text)
    if audio_data:
        print(f"Successfully retrieved audio data, length: {len(audio_data)} bytes.")
        # 如果你想在本地测试保存文件：
        # try:
        #     with open("test_output.wav", "wb") as f:
        #         f.write(audio_data)
        #     print("Test audio saved to test_output.wav. You can try playing this file.")
        # except IOError as io_err:
        #     print(f"Error saving test audio file: {io_err}")
    else:
        print("Failed to retrieve audio data from TTS.")