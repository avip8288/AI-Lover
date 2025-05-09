    # Dockerfile

    # 1. 选择一个 Python 基础镜像
    # 选择一个与你开发环境相近的 Python 版本
    FROM python:3.10-slim

    # 2. 设置工作目录
    # 在容器内创建一个目录用于存放应用代码
    WORKDIR /app

    # 3. 复制依赖文件
    # 将 requirements.txt 复制到工作目录
    # 先复制并安装依赖可以利用 Docker 的缓存机制，如果 requirements.txt 没有变化，则不需要重新安装
    COPY requirements.txt ./

    # 新增：安装编译 simpleaudio 所需的系统依赖
    RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libasound2-dev \
        && rm -rf /var/lib/apt/lists/*

    # 4. 安装依赖
    # --no-cache-dir 可以减少镜像大小
    RUN pip install --no-cache-dir -r requirements.txt

    # 5. 复制项目代码
    # 将项目根目录下的所有文件和文件夹复制到容器的 /app 目录
    # 注意：确保你的项目根目录是正确的，并且包含了 chat/, config/ 等子目录
    COPY . .

    # --- 添加这行 ---
    # 将 /app 目录添加到 Python 模块搜索路径
    ENV PYTHONPATH=/app

    # 6. 暴露端口
    # 告诉 Docker 容器将会监听哪个端口 (Streamlit 默认是 8501)
    EXPOSE 8501

    # 7. 定义容器启动时运行的命令
    # 使用 streamlit run 启动应用
    # --server.port 指定端口
    # --server.address=0.0.0.0 让 Streamlit 可以从容器外部访问
    CMD ["streamlit", "run", "chat/demo.py", "--server.port=8501", "--server.address=0.0.0.0"]