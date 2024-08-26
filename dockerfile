FROM python:3.11.0

RUN pip install --upgrade pip

WORKDIR /tmp

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . . 

#WORKDIR /tmp

#ARG WEBAPP_RELEASE
#RUN WEBAPP_RELEASE=$(git describe --tags $(git rev-list --tags #--max-count=1))

#ENV WEBAPP_RELEASE=$WEBAPP_RELEASE

WORKDIR /tmp/app

CMD sh -c 'WEBAPP_RELEASE=$(git describe --tags $(git rev-list --tags --max-count=1)) && python app.py'
