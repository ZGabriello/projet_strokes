FROM debian:latest

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip && pip3 install -r requirements.txt

RUN pip3 install pandas

COPY . /app

EXPOSE 5000

CMD python3 api.py
