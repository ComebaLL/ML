__author__ = "Kuvykin Nikita"

import gradio as gr
from typing import List
from module_download import EnvConfigProvider, PickleModelLoader
from model_unit import RegressionService, FormulaFormatter


class RegressionUI:
    """
    Отвечает только за отображение интерфейса.
    """
    def __init__(self, service):
        self._service = service                     # двигатель приводящий в работу программу
        self._interface = self._build_interface()   # руль который, позволяет управлять программой

    def _predict_wrapper(self, x1, x2, x3, x4):
        """
        Адаптер принимает данные от Gradio, передает в сервис,
        возвращает данные в формате, понятном Gradio.
        """
        try:
            pred, _, fig_scatter, fig_violin = self._service.process_prediction([x1, x2, x3, x4])
            return pred, fig_scatter, fig_violin
        except Exception as e:
            # Возвращаем ошибку в текстовое поле и None вместо графиков
            return f"Error: {str(e)}", None, None

    def _build_interface(self) -> gr.Interface:
        """
        Конфигурирует и собирает интерфейс.
        """
        
        # Получаем R2
        r2_score = self._service.get_model_quality()
        
        coefs = self._service._analyzer.get_coefficients_array()
        formula_latex = FormulaFormatter.format_linear_equation(coefs[1:], coefs[0]) if coefs is not None else "N/A"

        # 2. Описание UI
        description = f"""
        Введите значения 4 независимых переменных для получения предсказания.
        
        **Формула:** $${formula_latex}$$
        **R2 Score:** {r2_score}
        """


        interface = gr.Interface(
            fn=self._predict_wrapper,
            inputs=[
                gr.Number(label="X1 (Признак 1)"), 
                gr.Number(label="X2 (Признак 2)"),
                gr.Number(label="X3 (Признак 3)"), 
                gr.Number(label="X4 (Признак 4)")
            ],
            outputs=[
                gr.Textbox(label="Результат предсказания"),
                gr.Plot(label="Диаграмма рассеяния (Scatter)"),
                gr.Plot(label="Скрипичная диаграмма (Violin)")
            ],
            title="Предсказание Линейной Регрессии",
            description=description,
            allow_flagging="never" # Отключаем флаги для чистоты
        )
        return interface

    def launch(self, server_name="127.0.0.1", port=7860):
        """Запускает сервер."""
        self._interface.launch(server_name=server_name, server_port=port, share=False)


if __name__ == "__main__":
    try:
        config = EnvConfigProvider(env_path='model.env')
        loader = PickleModelLoader()
        
        path = config.get_model_path()
        loaded_model = loader.load(path)
        
        # Инициализируем сервис
        service = RegressionService(model=loaded_model, csv_path='lin_regres.csv')
        

        app = RegressionUI(service)
        
        print("Запуск сервера Gradio...")
        app.launch()
        
    except Exception as e:
        print(f"Критическая ошибка при запуске приложения: {e}")