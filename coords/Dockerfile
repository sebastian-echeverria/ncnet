FROM python:3.7

ENV PYTHONUNBUFFERED 1

# Install OpenCV and Gdal packages.
RUN apt-get update
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libgdal-dev gdal-bin
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Python deps.
RUN pip install --upgrade setuptools pip
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy app and set entry.
WORKDIR /app
COPY *.py /app/
ENTRYPOINT ["python", "projection.py"]
