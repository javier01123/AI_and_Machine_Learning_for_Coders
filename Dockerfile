FROM python:3.11

# Ensure logging is up to date despite possible buffering
# ENV PYTHONUNBUFFERED 1

WORKDIR /code
# COPY requirements.txt .
COPY src/ .

# Install Python dependencies only if requirements.txt has changed
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python","./script.py"]
