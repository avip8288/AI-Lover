# Dockerfile (位于 Project/sex_chat/ 目录下)

FROM python:3.10-slim

# 1. 设置应用在容器内的根目录，所有项目文件将直接放在这里
ENV APP_HOME=/app
WORKDIR ${APP_HOME} # WORKDIR 现在是 /app

# 2. 设置 PYTHONPATH，以便 Python 可以在 /app 下找到顶级模块/包
#    例如，如果 /app 下有 chat/ 和 Interaction_Design/ 目录，
#    代码中可以用 from chat import ... 或 from Interaction_Design import ...
ENV PYTHONPATH="${APP_HOME}"

# 3. 安装编译 simpleaudio 等所需的系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libasound2-dev && \
    rm -rf /var/lib/apt/lists/*

# 4. 复制 requirements.txt 并安装依赖
#    requirements.txt 位于构建上下文的根目录 (Project/sex_chat/)，会被复制到 /app/requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制项目的其余所有代码到 /app
#    现在，例如本地的 Project/sex_chat/chat/demo.py 会变成容器内的 /app/chat/demo.py
COPY . .

# 6. 暴露端口
EXPOSE 8501

# 7. 定义容器启动时运行的命令
#    由于 WORKDIR 是 /app，Streamlit 会在 /app 目录中寻找 chat/demo.py
#    即执行位于 /app/chat/demo.py 的脚本
CMD ["streamlit", "run", "chat/demo.py", "--server.port=8501", "--server.address=0.0.0.0"]