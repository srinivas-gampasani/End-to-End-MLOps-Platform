FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p outputs/screenshots
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "main.py"]
