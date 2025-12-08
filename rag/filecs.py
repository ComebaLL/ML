import os

def quote_blocks(input_file, output_file):
    try:
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Разделяем текст на блоки по двойному переносу строки 
        blocks = content.split('\n\n')

        quoted_blocks = []

        for block in blocks:

            clean_block = block.strip()
            
            # Если блок не пустой, оборачиваем его в кавычки
            if clean_block:
                quoted_blocks.append(f'"{clean_block}"')

        # Собираем блоки обратно в одну строку
        result_text = '\n\n'.join(quoted_blocks)

        # Записываем результат в новый файл
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"Готово! Результат записан в файл {output_file}")

    except FileNotFoundError:
        print(f"Ошибка: Файл {input_file} не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Запуск функции
quote_blocks('doc_skill_gems.txt', 'doc_gem1.txt')