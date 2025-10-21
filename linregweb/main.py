__author__ = "Kuvykin Nikita"

import gradio as gr
from module_download import load_model
from model_unit import load_training_data, get_r2_score, get_model_formula, get_top_features, create_scatter_plots, predict



# Загружаем модель
try:
    model = load_model()
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

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
        #gr.Image(label="Диаграмма рассеяния", height=400)
    ],
    title="Предсказание линейной регрессии",
    description=f"""
    Введите значения 4 независимых переменных для получения предсказания.
    
    **Модель:** {get_model_formula(model)} \n
    **R2 Score:** {get_r2_score(model)}
    
    """
)

# Запуск приложения
if __name__ == "__main__":
    regression_ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )