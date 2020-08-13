#!/bin/bash
docker build -t ocr_openvino .
docker stop ocr
docker rm ocr
docker run -p 8420:8000 -d --name ocr ocr_openvino
docker rmi $(docker images -qa -f 'dangling=true')
exit 0
