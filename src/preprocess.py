import os
import spacy
from typing import List

# Загрузка модели языка Spacy
# Для английского языка можно использовать 'en_core_web_lg'
# Для русского языка можно использовать 'ru_core_news_lg'
nlp = spacy.load("en_core_web_lg")

def read_text_files(directory: str) -> List[str]:
    """
    Читает все текстовые файлы из заданной директории и возвращает список текстов.
    
    :param directory: Путь к директории с .txt файлами
    :return: Список текстов, прочитанных из файлов
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def preprocess_text(text: str) -> str:
    """
    Выполняет базовую предобработку текста:
    - Удаление стоп-слов
    - Лемматизация (приведение слов к их начальной форме)
    - Удаление пунктуации
    
    :param text: Исходный текст для предобработки
    :return: Предобработанный текст
    """
    doc = nlp(text.lower())
    processed_text = []
    for token in doc:
        # Убираем стоп-слова, пунктуацию и сохраняем только леммы (начальные формы слов)
        if not token.is_stop and not token.is_punct:
            processed_text.append(token.lemma_)
    return " ".join(processed_text)

def process_all_texts(directory: str) -> List[str]:
    """
    Читает и предобрабатывает все текстовые файлы из указанной директории.
    
    :param directory: Путь к директории с .txt файлами
    :return: Список предобработанных текстов
    """
    texts = read_text_files(directory)
    processed_texts = [preprocess_text(text) for text in texts]
    return processed_texts
