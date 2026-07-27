FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей для OpenCV и MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода бота
COPY bot.py .

# Запуск
CMD ["python", "bot.py"]
