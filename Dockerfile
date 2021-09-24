FROM europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-6:latest

WORKDIR /usr/src/app

COPY ./trainer /usr/src/app/trainer
COPY ./setup.py /usr/src/app/setup.py

RUN python setup.py install
ENTRYPOINT python -m trainer.task
