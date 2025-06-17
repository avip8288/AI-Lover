# AI Lover Chatbot (AI伴侣聊天机器人)

[English](#english) | [中文](#中文)

---

## English

### 💘 Project Overview

This project is a sophisticated, interactive AI Lover designed to provide companionship and engaging, emotionally supportive conversations. She is not just a simple chatbot; she comes with features like long-term memory, voice, dynamic user profiling, and the ability to search the web for real-time information, allowing for a deeply personalized and immersive experience.

### ✨ Features

*   **Interactive Chat**: Engage in natural, empathetic, and multi-turn conversations.
*   **Voice Output**: Features real-time Text-to-Speech (TTS) powered by Cartesia, bringing the conversation to life.
*   **Long-Term Memory**: Remembers key details from your past conversations using a FAISS vector database to build a meaningful, lasting connection.
*   **Dynamic User Profiling**: Automatically creates and updates a user profile based on your chats to better understand your personality, preferences, and history.
*   **Web Search Integration**: Can access up-to-date information from the web via the Tavily API to discuss current events and a wide range of topics.
*   **Content Moderation**: Includes a safety guard layer to ensure conversations remain appropriate and constructive.
*   **Dynamic AI Emotional State (New!)**: Powered by a custom Mixture of Experts (MoE) model, the AI can dynamically perceive, express, and manage its own emotional state based on conversation context, leading to more human-like and empathetic interactions.
*   **Modular Tech Stack**: Built with a modern Python stack including Streamlit for the UI, LangGraph for the agentic logic, and a variety of powerful Large Language Models (LLMs) for different tasks.

### 🛠️ How to Use

#### 1. Clone the Repository
```bash
git clone https://github.com/tmracy/AI-Lover.git
cd AI-Lover # Changed: Navigate into the renamed project directory
```

#### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
```bash
python3 -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
# venv\Scripts\activate
```

#### 3. Install Dependencies
The project dependencies are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

#### 4. Configure API Keys and Models
The application requires API keys for various services.

1.  Navigate to the `config` directory. If it doesn't exist, create it:
    ```bash
    mkdir -p config
    ```
2.  Inside `config`, create a new file named `model_config.yaml`.
3.  Copy the template below into `model_config.yaml` and fill in your actual API keys, model names, and base URLs.

**File: `config/model_config.yaml`**
```yaml
model:
  Grok:
    model: 'llama3-70b-8192' # Example model name
    api_key: 'YOUR_GROK_API_KEY'
    api_base: 'https://api.groq.com/openai/v1'
  LLama:
    model: 'llama-guard-model-name' # Example model name for safety check
    api_key: 'YOUR_LLAMA_API_KEY'
    api_base: 'YOUR_LLAMA_API_BASE'
  DoubaoEmbedding:
    model: 'doubao-embedding-model' # Example embedding model
    api_key: 'YOUR_DOUBAO_API_KEY'
    api_base: 'YOUR_DOUBAO_API_BASE'
  Doubao:
    model: 'doubao-chat-model' # Example chat model
    api_key: 'YOUR_DOUBAO_API_KEY'
    api_base: 'YOUR_DOUBAO_API_BASE'
  Cartesia:
    url: 'https://api.cartesia.ai/tts/sse'
    voice_id: 'YOUR_CARTESIA_VOICE_ID' # Find this in your Cartesia dashboard
    api_key: 'YOUR_CARTESIA_API_KEY'
  Tavily:
    api_key: 'YOUR_TAVILY_API_KEY'
  Google:
    '2.5_flash_05-20': 'gemini-pro' # Example model name
    api_key: 'YOUR_GOOGLE_API_KEY'
    api_base: 'YOUR_GOOGLE_API_BASE'
  Qwen:
    model: 'qwen-turbo' # Example model name
    api_key: 'YOUR_QWEN_API_KEY'
    api_base: 'YOUR_QWEN_API_BASE'
  OpenAI:
    model: 'gpt-3.5-turbo' # Example model name
    api_key: 'YOUR_OPENAI_API_KEY'
    api_base: 'https://api.openai.com/v1'
```

#### 5. Configure Prompts
The AI's personality is defined by prompts in `config/prompt.yaml`. You must create this file for the application to work. You can customize the prompts to change the AI's behavior.

**File: `config/prompt.yaml`**
```yaml
Prompt:
  # Prompt to judge if a web search is needed
  judge: "Based on the user's query, do you need to search the internet for an answer? Respond with 'Yes' or 'No'."
  # Prompt to rewrite the user's query for better search results
  correction: "Rewrite the following user query to be more effective for a web search engine:"
  # System prompt for using search results
  use: "You have been given the following search results. Use them to answer the user's query."
  # Prompt to extract entities for the user profile
  entity: "From the following text, extract key entities like names, places, interests, etc., and return them as a JSON object."
  # Prompt to summarize a conversation snippet
  summary: "Summarize the key points of this conversation:"
  # Prompt to judge if a memory should be added to the user's profile
  judge_message: "Does this information contain a significant, long-term fact about the user that should be saved to their profile? Respond with 'Yes' or 'No'."
```

#### 6. Train the AI Emotion Model (Crucial Step for Emotional AI)
The AI's dynamic emotional state relies on a pre-trained Mixture of Experts (MoE) model. You need to train this model before running the main application.

1.  **Generate Training Data:** This step creates synthetic conversation data for training the MoE model.
    ```bash
    python -m Interaction_Design.MOE_data
    ```
    This will generate a `chat_train.json` file in the project root.

2.  **Train and Save the MoE Model:** This step trains the emotional model and saves it as `trained_moe_model.pth` in the project root.
    ```bash
    python -c "from Interaction_Design.AI_emotion_status import MOEModule, User; moe_module = MOEModule(User); moe_module.train_and_save_model(); print('Model training and saving initiated.')"
    ```
    *Note: This might take some time depending on your hardware and API limits.*

#### 7. Run the Application
The recommended way to run the app is through the Streamlit web interface.

```bash
streamlit run chat/demo.py
```
Open your web browser to the local URL provided by Streamlit. You will be asked to enter a session ID to start your conversation. Each unique session ID maintains its own separate memory and profile.

#### Alternative: Text-Only Mode (CLI)
If you prefer a command-line interface without the web UI and voice features, you can run the application directly in your terminal. This mode is ideal for quick tests or environments where a GUI is not available.

From within the `AI-Lover` project root directory, run:
```bash
python chat/main.py # Changed: Use direct script execution
```
The application will prompt you for a session ID and then you can start chatting directly in your terminal.

---

## 中文

### 💘 项目简介

本项目是一个高度互动的 AI 伴侣，旨在提供陪伴、情感支持和有吸引力的对话。她不只是一个简单的聊天机器人，她具备长期记忆、语音输出、动态用户画像和实时网络搜索等功能，为您带来深度个性化和沉浸式的交流体验。

### ✨ 功能特性

*   **互动对话**: 进行自然、富有同理心、可连续多轮的对话。
*   **语音输出**: 集成了由 Cartesia 驱动的实时文本转语音（TTS）功能，让对话栩栩如生。
*   **长期记忆**: 使用 FAISS 向量数据库记住你们过去对话的关键细节，建立一段有意义、持久的联系。
*   **动态用户画像**: 根据您的聊天内容，自动创建和更新用户档案，以更好地了解您的个性、偏好和经历。
*   **网络搜索集成**: 能通过 Tavily API 访问最新的网络信息，与您讨论时事和各种话题。
*   **内容审核**: 内置安全防护层，确保对话内容的适当性和建设性。
*   **动态AI情感状态 (New!)**: 由定制的混合专家（MoE）模型驱动，AI 能根据对话上下文动态感知、表达和管理自身的情绪状态，带来更拟人化、更具同理心的互动。
*   **模块化技术栈**: 使用现代 Python 技术栈构建，包括用于前端界面的 Streamlit、用于智能体逻辑的 LangGraph，以及用于不同任务的多种强大大型语言模型（LLM）。

### 🛠️ 如何使用

#### 1. 克隆代码库
```bash
git clone https://github.com/tmracy/AI-Lover.git
cd AI-Lover 
```

#### 2. 创建虚拟环境
强烈建议使用虚拟环境来管理项目依赖。
```bash
python3 -m venv venv
# 在 macOS/Linux 系统上
source venv/bin/activate
# 在 Windows 系统上
# venv\Scripts\activate
```

#### 3. 安装依赖
项目所需的依赖项已在 `requirements.txt` 文件中列出。
```bash
pip install -r requirements.txt
```

#### 4. 配置 API 密钥和模型
本应用需要多个服务的 API 密钥。

1.  进入 `config` 目录。如果它不存在，请创建它：
    ```bash
    mkdir -p config
    ```
2.  在 `config` 文件夹内，创建一个名为 `model_config.yaml` 的文件。
3.  将下面的模板内容复制到 `model_config.yaml` 中，并填入您自己的 API 密钥、模型名称和 API 地址。

**文件: `config/model_config.yaml`**
```yaml
model:
  Grok:
    model: 'llama3-70b-8192' # 模型名称示例
    api_key: '你的_GROK_API_KEY'
    api_base: 'https://api.groq.com/openai/v1'
  LLama:
    model: 'llama-guard-模型名称' # 安全检查模型名称示例
    api_key: '你的_LLAMA_API_KEY'
    api_base: '你的_LLAMA_API_BASE'
  DoubaoEmbedding:
    model: '豆包嵌入模型名称' # 嵌入模型名称示例
    api_key: '你的_DOUBAO_API_KEY'
    api_base: '你的_DOUBAO_API_BASE'
  Doubao:
    model: '豆包聊天模型名称' # 聊天模型名称示例
    api_key: '你的_DOUBAO_API_KEY'
    api_base: '你的_DOUBAO_API_BASE'
  Cartesia:
    url: 'https://api.cartesia.ai/tts/sse'
    voice_id: '你的_CARTESIA_VOICE_ID' # 在 Cartesia 官网后台查找
    api_key: '你的_CARTESIA_API_KEY'
  Tavily:
    api_key: '你的_TAVILY_API_KEY'
  Google:
    '2.5_flash_05-20': 'gemini-pro' # 模型名称示例
    api_key: '你的_GOOGLE_API_KEY'
    api_base: '你的_GOOGLE_API_BASE'
  Qwen:
    model: 'qwen-turbo' # 模型名称示例
    api_key: '你的_QWEN_API_KEY'
    api_base: '你的_QWEN_API_BASE'
  OpenAI:
    model: 'gpt-3.5-turbo' # 模型名称示例
    api_key: '你的_OPENAI_API_KEY'
    api_base: 'https://api.openai.com/v1'
```

#### 5. 配置提示词
AI 的性格和行为由 `config/prompt.yaml` 文件中的提示词定义。您必须创建此文件，否则程序将无法运行。您可以修改这些提示词来自定义 AI 的行为。

**文件: `config/prompt.yaml`**
```yaml
Prompt:
  # 用于判断是否需要网络搜索的提示词
  judge: "根据用户的提问，判断是否需要联网搜索才能回答？请只回答'是'或'否'。"
  # 用于改写用户问题以获得更好搜索结果的提示词
  correction: "请将以下用户的问题改写成一个更适合搜索引擎的查询语句："
  # 使用搜索结果进行回答的系统提示词
  use: "你获取了以下搜索结果，请利用这些信息来回答用户的问题。"
  # 用于为用户画像提取实体的提示词
  entity: "请从以下文本中提取关键实体，如姓名、地点、兴趣爱好等，并以JSON格式返回。"
  # 用于总结对话片段的提示词
  summary: "请总结这段对话的要点："
  # 用于判断是否应将信息存入用户档案的提示词
  judge_message: "这段信息是否包含关于用户的、值得长期保存的重要事实？请只回答'是'或'否'。"
```

#### 6. 训练AI情感模型 (AI情感核心步骤)
AI 的动态情感状态依赖于一个预训练的混合专家（MoE）模型。在运行主应用之前，您需要训练此模型。

1.  **生成训练数据：** 此步骤会为 MoE 模型生成合成对话数据。
    ```bash
    python -m Interaction_Design.MOE_data
    ```
    这将在项目根目录生成一个 `chat_train.json` 文件。

2.  **训练并保存MoE模型：** 此步骤将训练情感模型，并将其保存为项目根目录下的 `trained_moe_model.pth` 文件。
    ```bash
    python -c "from Interaction_Design.AI_emotion_status import MOEModule, User; moe_module = MOEModule(User); moe_module.train_and_save_model(); print('Model training and saving initiated.')"
    ```
    *注意：此过程可能需要一些时间，具体取决于您的硬件和API限制。*

#### 7. 运行程序
推荐通过 Streamlit Web 界面来运行此应用。

```bash
streamlit run chat/demo.py
```
在浏览器中打开 Streamlit 提供的本地网址。程序会要求您输入一个会话 ID 以开始对话。每个唯一的会话 ID 都会保留独立的聊天记忆和用户档案。

#### 备用选项：纯文本模式 (命令行)

如果您偏好使用命令行界面，不需要网页UI和语音功能，可以直接在终端中运行本应用。此模式非常适合快速测试或在没有图形界面的环境中使用。

在 `AI-Lover` 项目根目录下，运行：
```bash
python chat/main.py # 已修改：使用直接脚本执行方式
```
程序会提示您输入会话ID，之后您便可以直接在终端里开始聊天。

---