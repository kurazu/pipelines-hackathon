IMAGE_NAME=pipelines-hackathon-trainer

build: sdist
	docker build -t "${IMAGE_NAME}" .

run: build
	docker run \
		-it --rm \
		--name "${IMAGE_NAME}_run" \
		--volume "$(shell pwd)/../creds:/creds:ro" \
		--env GOOGLE_APPLICATION_CREDENTIALS="/creds/pipelines-hackathon-fa8dc6aa5fef.json" \
		--env AIP_MODEL_DIR="gs://pipelines-hackathon/models/simple-local/artifacts" \
		--env AIP_CHECKPOINT_DIR="gs://pipelines-hackathon/models/simple-local/checkpoints" \
		--env AIP_TENSORBOARD_LOG_DIR="gs://pipelines-hackathon/models/simple-local/tensorboard" \
		"${IMAGE_NAME}"

sdist:
	python setup.py sdist --formats=gztar

upload_trainer: sdist
	gsutil cp dist/trainer-0.1.tar.gz gs://pipelines-hackathon/trainer/trainer-0.1.tar.gz
