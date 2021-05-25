#!make

################################################################################
# Misc
################################################################################

clean:
# stop all containers
	docker stop $$(docker ps -a -q)

# destroy all containers
	docker rm --force $$(docker ps -a -q)

# destroy all images
	docker rmi --force $$(docker images -q)

# destroy all volumes
	docker volume rm $$(docker volume ls -q --filter dangling=true)

# remove if something is still dangling
	docker system prune