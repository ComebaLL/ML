__author__ = "Kuvykin Nikita"

import requests
from typing import Dict, Any, Optional

class RegressionClient:
    """
    Класс-клиент для взаимодействия с API модели регрессии.
    """
    def __init__(self, base_url: str, timeout: int = 5):

        # Убираем слеш в конце, если он есть, для корректной склейки путей
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """
        Внутренний метод для выполнения GET запросов.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Не удалось подключиться к серверу: {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Ошибка API ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка при запросе к {url}: {e}")

    def ping_server(self) -> bool:
        """Проверяет доступность сервера."""
        try:
            data = self._get("/ping")
            return data.get("status") == "ok"
        except Exception:
            return False

    def get_prediction(self, x1: float, x2: float, x3: float, x4: float) -> Dict[str, Any]:
        """Получает предсказание модели по 4 признакам."""
        params = {
            "x1": x1, "x2": x2, "x3": x3, "x4": x4
        }
        return self._get("/predict", params=params)

    def get_metrics(self) -> Dict[str, Any]:
        """Получает метрики модели."""
        return self._get("/metrics")


def main():
    # Настройки
    API_URL = "http://127.0.0.1:8000/api/v1"
    
    # Создаем экземпляр клиента
    client = RegressionClient(base_url=API_URL)
    
    print(f"Подключение к {API_URL}...\n")

    # Проверка связи
    if client.ping_server():
        print("PING: Сервер доступен и готов к работе.")
    else:
        print("PING: Сервер недоступен. Завершение работы.")
        return

    print("-" * 30)

    # Запрос предсказания
    print("Отправка запроса на предсказание")
    try:
        # Пример данных
        features = [0.5, 1.2, 3.0, 0.1]
        
        result = client.get_prediction(*features)
        
        print(f"PREDICT Результат: {result['prediction']}")
        print(f"Полные данные ответа: {result}")
        
    except Exception as e:
        print(f"PREDICT Ошибка: {e}")

    print("-" * 30)

    # 4. Запрос метрик
    print("Запрос метрик модели...")
    try:
        metrics = client.get_metrics()
        print(f"METRICS Данные: {metrics}")
    except Exception as e:
        print(f"METRICS Ошибка: {e}")


if __name__ == "__main__":
    main()