FROM python:3.11.0

RUN pip install --upgrade pip

WORKDIR /tmp

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt
ARG MLCODE_RELEASE
RUN pip install git+"https://github.com/noaa-afsc/ftnirs-ml-codebase.git@${MLCODE_RELEASE}"

COPY . . 

WORKDIR /tmp/app

RUN mkdir -p ./tmp

#pull from latest commit in checkout branch
RUN WEBAPP_RELEASE=$(git describe --tags $(git rev-list --tags --max-count=1 --first-parent)) && echo "WEBAPP_RELEASE=${WEBAPP_RELEASE}" > ./tmp/.dynenv

CMD ["python","app.py"]
