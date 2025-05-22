docker build -t sex-chat-app_v0.0.3 .
docker run -d -p 8501:8501 --name sex-chat-container sex-chat-app_v0.0.3
docker start sex-chat-container
docker ps
docker stop sex-chat-container
docker rm sex-chat-container
docker exec sex-chat-container tail -n 200 logs/sex_log.log


docker tag sex-chat-app_v0.0.1 <你的DockerHub用户名>/sex-chat-app:v0.0.1
        # 把 <你的DockerHub用户名> 换成你真实的用户名

'''
接下来说一下这几个操作的一些本质部分
docker build部分，这里是平台切换到了linux上，sex-chat-app是镜像名称，v1.0.1是标签

docker tag 是打标签，其中sex-chat-app:v1.0.1是部分必须和docker build中指定的镜像标签一致
后面的tmracy是仓库或者是用户，这个不能自定义，要保证存在的
而tmracy/sex-chat-app:v1.0.1中的sex-chat-app:v1.0.1则是可以自定义命名，这个是镜像标签的别名

docker push这边的tmracy/sex-chat-app:v1.0.要和docker tag打的镜像标签的别名一致

'''
# 打标签
docker tag sex-chat-app:v1.0.2 tmracy/sex-chat-app:v1.0.2

# 推送
docker push tmracy/sex-chat-app:v1.0.2

docker pull --platform linux/amd64 python:3.10-slim
#linux/amd64架构
docker build --platform linux/amd64 -t sex-chat-app:v1.0.2 .