# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /code

# Install system dependencies
RUN apt-get update \
  && apt-get -y install netcat-openbsd gcc libgl1-mesa-glx libglib2.0-0 \
     ffmpeg libsm6 libxext6 libxrender-dev libgtk2.0-dev libpng-dev libjpeg-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /code/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /code
COPY . /code/

# Command to run the application
CMD exec uvicorn app.main:app --workers 1 --timeout-keep-alive 0 --port 8080 --host 0.0.0.0 --reload
