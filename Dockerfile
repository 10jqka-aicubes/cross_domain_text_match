FROM hub-dev.hexin.cn/jupyterhub/nvidia_cuda:py37-cuda100-ubuntu18.04-v2

COPY ./ /home/jovyan/cross_domain_text_match 

RUN cd /home/jovyan/cross_domain_text_match  && \
    python -m pip install -r requirements.txt 