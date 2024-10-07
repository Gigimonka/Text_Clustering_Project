import os
import spacy
from typing import List
from num2words import num2words
import re
import html


# Загрузка модели языка Spacy
# Для английского языка можно использовать 'en_core_web_sm'
# Для русского языка можно использовать 'ru_core_news_sm'
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

def clean_text(text: str) -> str:
    """
    Очищает текст с выполнением следующих шагов:
    - Удаление HTML-тегов
    - Удаление URL-адресов
    - Удаление нерелевантных символов
    - Удаление пунктуации
    - Удаление дублирующегося текста
    - Удаление чисел или их замена на текст
    - Обработка пробелов
    
    :param text: Исходный текст для очистки
    :return: Очищенный текст
    """
    # 1. Удаляем HTML-теги
    text = html.unescape(text)
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Удаляем URL-адреса
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Убираем нерелевантные символы (оставляем только буквы, цифры и пробелы)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 4. Убираем дублирующийся текст (используем уникальный набор слов)
    words = text.split()
    unique_words = list(dict.fromkeys(words))  # Убираем дубликаты
    text = " ".join(unique_words)
    
    # 5. Замена чисел на текст
    text = re.sub(r'\d+', lambda x: num2words(int(x.group())), text)
    
    # 8. Обрабатываем пробелы (удаление лишних пробелов)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_text(text: str) -> str:
    """
    Выполняет базовую предобработку текста:
    - Удаление стоп-слов
    - Лемматизация (приведение слов к их начальной форме)
    - Удаление пунктуации
    
    :param text: Исходный текст для предобработки
    :return: Предобработанный текст
    """
    text = clean_text(text)

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
    print(processed_texts)
    return processed_texts

if __name__== "__main__":
    process_all_texts("data/")