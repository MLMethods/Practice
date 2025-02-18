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
    --volume ./Repos:/home/ubuntu/Repos \
    ml-student-image

sync:
	docker exec ml-student bash -c "cd /home/ubuntu/Repos/mlmethods/Practice && git pull"
	docker exec ml-student bash -c "cd /home/ubuntu/Repos/mlmethods/Assignments && git pull"

exec: 
	docker exec -it ml-student bash

rm:
	docker rm -f ml-student
