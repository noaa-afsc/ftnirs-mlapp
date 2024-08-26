FROM python:3.11.0

ENV WEBAPP_RELEASE $(git describe --tags $(git rev-list --tags --max-count=1))

RUN pip install --upgrade pip

WORKDIR /tmp

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . . 

WORKDIR /app

CMD ["python","app.py"]
