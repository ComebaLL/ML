__author__ = "Kuvykin Nikita"

import gradio as gr
from module_download import load_model
from model_unit import get_r2_score, get_model_coefficients, predict, coefficients_for_formul


# Загружаем модель
try:
    model = load_model()
    print("Модель загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None


r2_score_value = get_r2_score(model)
# Получаем коэффициенты как numpy array
coef_array = get_model_coefficients(model)

# Генерируем LaTeX формулу из коэффициентов
latex_formula = coefficients_for_formul(coef_array)


# Создаем основной интерфейс
regression_ui = gr.Interface(
    fn=lambda x1, x2, x3, x4: predict(model, x1, x2, x3, x4),
    inputs=[
        gr.Number(label="X1"), 
        gr.Number(label="X2"),
        gr.Number(label="X3"), 
        gr.Number(label="X4")
    ],
    outputs=[
        gr.Textbox(label="Результат"),
        gr.Plot(label="Диаграмма рассеяния")
    ],
    title="Предсказание линейной регрессии",
    description=f"""
    Введите значения 4 независимых переменных для получения предсказания.
    
    **Формула:** $${latex_formula}$$ \n
    **R2 Score:** {r2_score_value}
    
    """
)

# Запуск приложения
if __name__ == "__main__":
    regression_ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )