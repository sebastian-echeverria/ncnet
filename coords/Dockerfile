FROM python:3.7

ENV PYTHONUNBUFFERED 1

# Install OpenCV and Gdal packages.
RUN apt-get update
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y gdal-bin
#binutils libproj-dev

# Install Python deps.
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt

# Copy app and set entry.
WORKDIR /app
COPY *.py /app/
ENTRYPOINT ["python", "projection.py"]
