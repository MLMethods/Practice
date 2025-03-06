# Commands

IMAGE := ml-student-image
CONTAINTER := ml-student

build:
	docker build --tag $(IMAGE) --file Dockerfile .

run:
	docker run \
    --rm \
    --interactive \
    --tty \
    --publish 8888:8888 \
    --name $(CONTAINTER) \
    --volume ./Repos:/home/ubuntu/Repos \
    $(IMAGE)

sync:
	docker exec $(CONTAINTER) bash -c "cd /home/ubuntu/Repos/mlmethods/Practice && git pull"
	docker exec $(CONTAINTER) bash -c "cd /home/ubuntu/Repos/mlmethods/Assignments && git pull"

exec:
	docker exec -it $(CONTAINTER) bash

rm:
	docker rm -f $(CONTAINTER)

rmi:
	docker rmi -f $(IMAGE)
