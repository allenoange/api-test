# docker run -it --runtime=nvidia --network=host --name test mindspore_gpu:1.3.0

FROM mindspore/mindspore-gpu:1.2.1
WORKDIR /app
RUN yes | pip install --upgrade pip
# RUN yes | pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101  -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
RUN yes | pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.3.0/MindSpore/gpu/x86_64/cuda-10.1/mindspore_gpu-1.3.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN yes | pip install tensorflow==2.4.1
RUN yes | pip install scipy==1.5.3  # upgrade scipy to satisfy mindinsight and mindspore requirement
RUN yes | pip install mindinsight
RUN yes | pip install onnx==1.9.0
RUN yes | pip install onnxconverter-common
RUN yes | pip install onnxoptimizer
RUN yes | pip install onnxruntime
RUN yes | pip install pandas
RUN yes | pip install matplotlib
RUN yes | pip install opencv-python
RUN yes | pip install netron
RUN yes | pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101  -f https://download.pytorch.org/whl/torch_stable.html
RUN yes | pip install notebook                          
