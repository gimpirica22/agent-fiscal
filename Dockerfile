FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_fiscal.py .
COPY bot_telegram.py .

CMD ["python", "-u", "bot_telegram.py"]
