from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List
import pickle
from scipy.spatial.distance import cdist
from src.visualization import visualize_elbow_and_silhouette
import numpy as np

def find_optimal_clusters(embeddings: List[List[float]], max_k: int = 10) -> None:
    """
    Использует метод локтя и силуэтный анализ для определения оптимального количества кластеров.
    
    :param embeddings: Список эмбеддингов
    :param max_k: Максимальное количество кластеров для оценки
    """
    distortions, inertias, silhouette_scores = [], [], []
    K = range(2, max_k + 1)

    for k in K:
        # Обучаем модель
        kmeans = cluster_embeddings(embeddings, n_clusters=k)

        # Вычисляем distortion, inertia и силуэтный коэффициент
        distortions.append(sum(np.min(cdist(embeddings, kmeans.cluster_centers_, 'euclidean'), axis=1)) / len(embeddings))
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))

    # Определение оптимального K по максимальному силуэтному коэффициенту
    max_silhouette_score = max(silhouette_scores)
    optimal_k = K[silhouette_scores.index(max_silhouette_score)]
    
    # Вызов функции для визуализации
    visualize_elbow_and_silhouette(K, distortions, inertias, silhouette_scores, optimal_k, max_silhouette_score)

def cluster_embeddings(embeddings: List[List[float]], n_clusters: int) -> KMeans:
    """
    Кластеризует эмбеддинги с использованием KMeans.
    
    :param embeddings: Список эмбеддингов
    :param n_clusters: Количество кластеров
    :return: Обученная модель KMeans
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans

def save_model(kmeans_model: KMeans, filepath: str) -> None:
    """
    Сохраняет модель KMeans в файл.
    
    :param kmeans_model: Обученная модель KMeans
    :param filepath: Путь для сохранения модели
    """
    with open(filepath, 'wb') as f:
        pickle.dump(kmeans_model, f)

def load_model(filepath: str) -> KMeans:
    """
    Загружает модель KMeans из файла.
    
    :param filepath: Путь к файлу с моделью
    :return: Загруженная модель KMeans
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
