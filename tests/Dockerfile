FROM debian:latest

COPY requirements.txt /tests/requirements.txt

WORKDIR /tests

RUN apt-get update && apt-get install -y python3-pip && pip3 install -r requirements.txt

COPY . /tests

CMD  python3 -m unittest test_api.py
