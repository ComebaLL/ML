FROM python:3.12

WORKDIR .

COPY . .
COPY reqrmt.txt .
COPY model/ ./model/

# Обновляем pip перед установкой
RUN pip install --no-cache-dir -r reqrmt.txt

# Запускаем приложение с указанием хоста и порта
CMD ["python", "main.py"]