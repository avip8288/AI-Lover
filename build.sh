docker build -t sex-chat-app_v0.0.3 .
docker run -d -p 8501:8501 --name sex-chat-container sex-chat-app_v0.0.3
docker start sex-chat-container
docker ps
docker stop sex-chat-container
docker rm sex-chat-container
docker exec sex-chat-container tail -n 200 logs/sex_log.log


docker tag sex-chat-app_v0.0.1 <你的DockerHub用户名>/sex-chat-app:v0.0.1
        # 把 <你的DockerHub用户名> 换成你真实的用户名

# 打标签
docker tag sex-chat-app_v0.0.5 tmracy/sex-chat-app:v0.0.5

# 推送
docker push tmracy/sex-chat-app:v0.0.5

docker pull --platform linux/amd64 python:3.10-slim
#linux/amd64架构
c