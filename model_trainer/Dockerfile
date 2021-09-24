FROM europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-6:latest

WORKDIR /app

COPY dist/trainer-0.1.tar.gz /app/trainer-0.1.tar.gz
RUN pip3 install --user --upgrade --force-reinstall --no-deps trainer-0.1.tar.gz

ENTRYPOINT python -m trainer.task
