# syntax=docker/dockerfile:1
FROM jupyter/scipy-notebook
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /home/jovyan/work
USER root
COPY run_script.sh run_script.sh
RUN chmod +x run_script.sh
USER jovyan
CMD ["./run_script.sh"]
