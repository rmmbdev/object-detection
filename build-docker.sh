docker build --shm-size 4gb --build-arg http_proxy=http://192.168.1.132:8118 --build-arg https_proxy=http://192.168.1.132:8118 --tag=obj-04 .