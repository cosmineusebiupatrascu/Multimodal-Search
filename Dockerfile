FROM ubuntu:latest
LABEL authors="Cosmin"

ENTRYPOINT ["top", "-b"]