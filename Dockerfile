FROM python:3.9.13-slim-buster

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && \
    apt-get install python3-opencv -y

RUN apt-get install g++  -y && \
    apt-get install gunicorn  -y && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "gunicorn", "-w 4", "-b 0.0.0.0", "app:app" ]