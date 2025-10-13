FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install python and its dependencies
RUN apt update && \
    apt install -y --no-install-recommends curl python3 python3-pip python3-venv

# Create a new venv
RUN python3 -m venv /venvs/litert-demo

# Install dependencies (in the venv)
COPY requirements.txt ./
RUN . /venvs/litert-demo/bin/activate && \
    pip3 install -r requirements.txt

# Copy the rest of the application
COPY . ./
