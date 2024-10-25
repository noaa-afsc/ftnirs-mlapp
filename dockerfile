FROM python:3.11.0

RUN pip install --upgrade pip

WORKDIR /tmp

COPY requirements.txt .
COPY .env .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

#do this seperately- maybe better to include in requirements.txt, but slightly more complex to edit within the file
ARG MLCODE_RELEASE
RUN export MLCODE_RELEASE=$(grep MLCODE_RELEASE .env | cut -d '=' -f2) && \
pip install git+https://github.com/DanWoodrichNOAA/ftnirs-ml-codebase.git@$VERSION_TAG

COPY . . 

WORKDIR /tmp/app

RUN mkdir -p ./tmp

#pull from latest commit in checkout branch
RUN WEBAPP_RELEASE=$(git describe --tags $(git rev-list --tags --max-count=1 --first-parent)) && echo "WEBAPP_RELEASE=${WEBAPP_RELEASE}" > ./tmp/.dynenv

CMD ["python","app.py"]
