FROM python:3.11.0

RUN pip install --upgrade pip

WORKDIR /tmp

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . . 
ARG GIT_TAG
RUN GIT_TAG=$(git describe --tags $(git rev-list --tags --max-count=1))

ENV GIT_TAG $GIT_TAG

WORKDIR /tmp/app

CMD ["python","app.py"]
