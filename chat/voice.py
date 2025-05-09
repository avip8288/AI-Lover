import requests
import yaml
# import simpleaudio as sa # simpleaudio 不再需要在服务器端播放
import io
import os
import json # For parsing SSE data
import base64 # For decoding audio chunks
import struct # For a local test in __main__ to build WAV

# Get the directory of the current script (voice.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one directory to get to the 'chat' parent directory (which should be 'sex_chat')
project_chat_dir = os.path.dirname(current_script_dir)
# Construct the path to the config file
config_file_path = os.path.join(project_chat_dir, 'config', 'model_config.yaml')

with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

url = config['model']['Cartesia']['url'] # Should be https://api.cartesia.ai/tts/sse
voice_id = config['model']['Cartesia']['voice_id']
api_key = config['model']['Cartesia']['api_key']

def TTS(text: str):
    api_output_format = {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 44100,
    }
    payload = {
        "model_id": "sonic-2",
        "transcript": text,
        "language": "zh",
        "voice": {"mode": "id", "id": voice_id},
        "output_format": api_output_format,
        "speed": "slow"
    }
    headers = {
        "Cartesia-Version": "2025-04-16",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30, stream=True)
        print(f"SSE: Status Code from Cartesia API: {response.status_code}")
        response.raise_for_status()

        chunk_yielded_count = 0
        for line_bytes in response.iter_lines():
            if line_bytes:
                decoded_line = line_bytes.decode('utf-8')
                # print(f"SSE Raw Line: {decoded_line}") 

                if decoded_line.startswith('data:'):
                    event_data_str = decoded_line[len('data: '):].strip()
                    # print(f"SSE Data String: \"{event_data_str}\"") 

                    if not event_data_str:
                        # print("SSE: Data String is empty, skipping.")
                        continue
                    try:
                        event_data = json.loads(event_data_str)
                        # print(f"SSE Parsed JSON Data: {event_data}") 

                        event_type = event_data.get('type')

                        if event_type == 'chunk':
                            audio_data_base64 = event_data.get('data')
                            if audio_data_base64:
                                try:
                                    audio_chunk = base64.b64decode(audio_data_base64)
                                    yield audio_chunk
                                    chunk_yielded_count += 1
                                    # print(f"SSE: Yielded chunk {chunk_yielded_count}, size {len(audio_chunk)}")
                                except base64.binascii.Error as b64_err:
                                    print(f"SSE: Base64 decoding error for 'data' field in 'chunk' event: {b64_err}")
                                except Exception as e_dec:
                                     print(f"SSE: Error decoding 'data' field in 'chunk' event: {e_dec}")
                            else:
                                print("SSE: 'chunk' event received, but 'data' field is missing or null.")
                        
                        elif event_type == 'done':
                            print(f"SSE: Received 'done' event: {event_data}")
                            break 
                        
                        elif event_type == 'flush_done':
                            print(f"SSE: Received 'flush_done' event: {event_data}")
                            if event_data.get('done') is True:
                                print("SSE: 'flush_done' event also indicates overall completion.")
                                break
                        
                        elif event_type == 'timestamps' or event_type == 'phoneme_timestamps':
                            pass 

                        elif event_type == 'error': 
                            print(f"SSE: Received 'error' event: {event_data.get('message', event_data)}")
                            break
                        
                        else:
                            print(f"SSE: Received unknown event type or unhandled event: {event_data}")

                    except json.JSONDecodeError as json_err:
                        print(f"SSE: JSONDecodeError: {json_err} for data string: \"{event_data_str}\"")
                    except Exception as e_proc:
                        print(f"SSE: Error processing SSE line/event: {e_proc} - Line: {decoded_line}", exc_info=True)
                # else:
                #     if decoded_line.strip():
                #         print(f"SSE Other Line: {decoded_line}")
        
        print(f"SSE: Finished processing stream. Total chunks yielded: {chunk_yielded_count}")

    except requests.exceptions.HTTPError as http_err:
        print(f"SSE: Cartesia API HTTP error occurred: {http_err} - Response: {response.text if 'response' in locals() and response else 'No response object'}")
    except requests.exceptions.RequestException as req_err:
        print(f"SSE: Request to Cartesia API failed: {req_err}")
    except Exception as e:
        print(f"SSE: TTS function encountered an unknown error: {e}", exc_info=True)

if __name__ == '__main__':
    sample_text = "你好，这是一个流式语音测试。"
    print(f"Testing TTS generator with: '{sample_text}'")
    
    accumulated_data_len = 0
    all_chunks = bytearray()
    for i, chunk in enumerate(TTS(sample_text)):
        if chunk:
            all_chunks.extend(chunk)
            accumulated_data_len += len(chunk)
            print(f"MainTest: Received chunk {i+1}, size: {len(chunk)}, total accumulated: {accumulated_data_len}")
        else:
            print("MainTest: Received an empty or None chunk.")
    
    if accumulated_data_len > 0:
        print(f"MainTest: Finished consuming generator. Total raw data bytes: {accumulated_data_len}")
        
        def _create_wav_header_local(sample_rate: int, num_channels: int, sample_width_bytes: int, num_frames: int) -> bytes:
            datasize = num_frames * num_channels * sample_width_bytes
            header = bytearray()
            header.extend(b'RIFF'); header.extend(struct.pack('<I', datasize + 36))
            header.extend(b'WAVE'); header.extend(b'fmt '); header.extend(struct.pack('<I', 16))
            header.extend(struct.pack('<H', 1)); header.extend(struct.pack('<H', num_channels))
            header.extend(struct.pack('<I', sample_rate)); header.extend(struct.pack('<I', sample_rate * num_channels * sample_width_bytes))
            header.extend(struct.pack('<H', num_channels * sample_width_bytes)); header.extend(struct.pack('<H', sample_width_bytes * 8))
            header.extend(b'data'); header.extend(struct.pack('<I', datasize))
            return bytes(header)

        if all_chunks:
            sr, nc, swb = 44100, 1, 2
            num_f = len(all_chunks) // (nc * swb)
            if num_f > 0:
                header = _create_wav_header_local(sr, nc, swb, num_f)
                with open("test_direct_sse_output.wav", "wb") as f:
                    f.write(header + all_chunks)
                print("MainTest: Saved concatenated raw data with a header to test_direct_sse_output.wav")
            else:
                print("MainTest: Not enough data to form a WAV file for saving.")
    else:
        print("MainTest: TTS generator did not yield any data.")