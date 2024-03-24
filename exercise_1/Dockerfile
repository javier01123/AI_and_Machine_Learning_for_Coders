FROM python:3.11

# Ensure logging is up to date despite possible buffering
ENV PYTHONUNBUFFERED 1

WORKDIR /code
# COPY requirements.txt .
COPY src/ .

RUN pip install -r requirements.txt 

# CMD ["python","./script.py"]
