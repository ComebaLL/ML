FROM python:3.10-slim

# Установка рабочей директории в контейнере
WORKDIR /app

# Копирование файла зависимостей в контейнер
COPY . /app

# Установка зависимостей
RUN pip install --no-cache-dir -r reqrmt.txt

# Порт для FastAPI приложения
EXPOSE 8000

# Запуск uvicorn сервера для FastAPI приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
