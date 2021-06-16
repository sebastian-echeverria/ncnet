FROM python:3.7

RUN pip install pipenv

# Installing Python deps without a venv (not needed in container).
WORKDIR /app
COPY Pipfile /app
COPY Pipfile.lock /app
#RUN pipenv install --system --deploy --ignore-pipfile
RUN pipenv lock --keep-outdated --requirements > requirements.txt
RUN pip install -r requirements.txt

# Actual code.
WORKDIR /app
COPY *.py /app
COPY lib lib/
ENTRYPOINT ["python3", "train.py"]
