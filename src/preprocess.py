import os
import spacy
from typing import List
import re
import html

# Загрузка модели SpaCy
nlp = spacy.load("en_core_web_lg")

def read_text_files(directory: str) -> List[str]:
    """
    Читает все текстовые файлы из заданной директории и возвращает список текстов.
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def clean_text(text: str) -> str:
    """
    Очищает текст от HTML-тегов и URL-адресов, а также убирает лишние пробелы.
    
    :param text: Исходный текст для очистки
    :return: Очищенный текст
    """
    # Удаляем HTML-теги и URL-адреса
    text = html.unescape(text)
    text = re.sub(r'<.*?>', '', text)  # Удаление HTML-тегов
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Удаление URL
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text_spacy(text: str) -> str:
    """
    Выполняет предобработку текста с использованием SpaCy:
    - Токенизация
    - Лемматизация
    - Удаление стоп-слов
    - Удаление пунктуации
    
    :param text: Исходный текст для предобработки
    :return: Предобработанный текст
    """
    # Чистим текст от HTML и лишних символов
    text = clean_text(text)
    
    # Пропускаем текст через SpaCy для обработки
    doc = nlp(text)
    
    # Формируем список лемм, исключая стоп-слова и пунктуацию
    processed_text = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(processed_text)

def process_all_texts(directory: str) -> List[str]:
    """
    Читает и предобрабатывает все текстовые файлы из указанной директории с использованием SpaCy.
    
    :param directory: Путь к директории с .txt файлами
    :return: Список предобработанных текстов
    """
    texts = read_text_files(directory)
    processed_texts = [preprocess_text_spacy(text) for text in texts]
    print(processed_texts)
    return processed_texts

if __name__ == "__main__":
    process_all_texts("data/")
