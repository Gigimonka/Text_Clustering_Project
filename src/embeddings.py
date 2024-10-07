from sentence_transformers import SentenceTransformer
from typing import List
import pickle

# Загружаем модель из библиотеки sentence-transformers
# Можно выбрать любую модель, например 'all-MiniLM-L6-v2'
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Преобразует список текстов в эмбеддинги с использованием модели Sentence-Transformers.
    
    :param texts: Список предобработанных текстов
    :return: Список эмбеддингов
    """
    embeddings = model.encode(texts)
    return embeddings

def save_embeddings(embeddings: List[List[float]], filepath: str) -> None:
    """
    Сохраняет эмбеддинги в файл для последующего использования.
    
    :param embeddings: Список эмбеддингов
    :param filepath: Путь для сохранения файла с эмбеддингами
    """
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filepath: str) -> List[List[float]]:
    """
    Загружает сохраненные эмбеддинги из файла.
    
    :param filepath: Путь к файлу с эмбеддингами
    :return: Список эмбеддингов
    """
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings
