FROM python:3.12

# Установка рабочей директории в контейнере
WORKDIR /app

# Копирование файла зависимостей в контейнер
COPY reqrmt.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r reqrmt.txt

# Копирование приложения и модели в контейнер
COPY app/ ./app/
COPY model/ ./model/

# Порт для FastAPI приложения
EXPOSE 8000

# Запуск uvicorn сервера для FastAPI приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]