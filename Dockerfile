FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Set up pytho, pip, and pipenv.
RUN apt-get update
RUN apt-get -y install python3.7 python3-pip python3.7-dev
RUN pip3 install pipenv

# Installing Python deps without a venv (not needed in container).
WORKDIR /app
#COPY Pipfile /app
#COPY Pipfile.lock /app
#RUN pipenv install --system --deploy --ignore-pipfile
#ENV LC_ALL=en_US.utf-8
#ENV export LANG=en_US.utf-8
#RUN pipenv lock --keep-outdated --requirements > requirements.txt
COPY requirements.txt /app
RUN pip3 install -r requirements.txt

# Actual code.
WORKDIR /app
COPY *.py /app/
COPY lib lib/
ENTRYPOINT ["python3.7", "train.py"]
