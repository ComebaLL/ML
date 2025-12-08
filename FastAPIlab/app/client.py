import requests

# Адрес сервера
BASE_URL = "http://127.0.0.1:8000/api/v1"

print(f"Подключаемся к {BASE_URL}...\n")

# Проверка работы сервера
try:
    response = requests.get(f"{BASE_URL}/ping")
    print("PING Ответ:", response.status_code)
    print("PING Тело:", response.json())
except Exception as e:
    print("Сервер недоступен.")
    exit()

print("-" * 30)

# Получение предсказания
my_params = {
    "x1": 0.5,
    "x2": 1.2,
    "x3": 3.0,
    "x4": 0.1
}

response = requests.get(f"{BASE_URL}/predict", params=my_params)

if response.status_code == 200:
    data = response.json()
    print(f"PREDICT Результат: {data['prediction']}")
    print(f"PREDICT Полные данные: {data}")
else:
    print("PREDICT Ошибка:", response.text)

print("-" * 30)

# Получение метрик
response = requests.get(f"{BASE_URL}/metrics")

if response.status_code == 200:
    print("METRICS Данные:", response.json())
else:
    print("METRICS Ошибка:", response.text)