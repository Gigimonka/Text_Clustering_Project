from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def classify_new_document(new_embedding: List[float], cluster_embeddings: List[List[float]]) -> int:
    """
    Классифицирует новый документ на основе его эмбеддинга, находя ближайший кластер
    с использованием косинусного расстояния.
    
    :param new_embedding: Эмбеддинг нового документа
    :param cluster_embeddings: Эмбеддинги центроидов кластеров
    :return: Номер ближайшего кластера
    """
    similarities = cosine_similarity([new_embedding], cluster_embeddings)
    closest_cluster = similarities.argmax()
    return closest_cluster

def find_nearest_neighbors(new_embedding: List[float], embeddings: List[List[float]], n_neighbors: int = 1) -> List[int]:
    """
    Находит ближайшие эмбеддинги (и их кластеры) для нового документа с использованием k-ближайших соседей.
    
    :param new_embedding: Эмбеддинг нового документа
    :param embeddings: Список эмбеддингов обучающих данных
    :param n_neighbors: Количество соседей, которые необходимо найти
    :return: Список индексов ближайших соседей
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors([new_embedding])
    return indices.flatten()
