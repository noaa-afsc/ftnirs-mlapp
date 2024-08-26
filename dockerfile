FROM python:3.11.0

ENV WEBAPP_RELEASE $(git describe --tags $(git rev-list --tag
s --max-count=1))

WORKDIR /tmp

RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app.py /tmp/app.py

WORKDIR /tmp

CMD ["python","app.py"]
