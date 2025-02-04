include .env
export $(shell sed 's/=.*//' .env)

# Commands

build:
	docker build --tag ml-student-image --file Dockerfile .

run:
	docker run \
    --rm \
    --interactive \
    --tty \
    --publish 8888:8888 \
    --name ml-student \
    --volume $(BASE_DIR):$(BASE_DIR) \
    ml-student-image

sync:
	docker exec ml-student bash -c "cd $(PRACTICE_REPO) && git pull"
	docker exec ml-student bash -c "cd $(ASSIGNMENTS_REPO) && git pull"

rm:
	docker rm -f ml-student
