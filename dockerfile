# 基础镜像
FROM python:3.11

WORKDIR /Code/

RUN apt-get -y update

COPY /face-anti-spoofing/ /Code/

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple


# 暴露端口
EXPOSE 8100

CMD ["python", "AppStart.py"]