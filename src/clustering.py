from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List
import pickle

def find_optimal_clusters(embeddings: List[List[float]], max_k: int = 10) -> None:
    """
    Использует метод локтя для определения оптимального количества кластеров.
    
    :param embeddings: Список эмбеддингов
    :param max_k: Максимальное количество кластеров для оценки
    """
    distortions = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, max_k+1), distortions, marker='o')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Инерция (Distortion)')
    plt.title('Метод локтя для определения оптимального количества кластеров')
    plt.show()

def silhouette_analysis(embeddings: List[List[float]], max_k: int = 10) -> None:
    """
    Использует силуэтный коэффициент для оценки качества кластеризации при различных значениях K.
    
    :param embeddings: Список эмбеддингов
    :param max_k: Максимальное количество кластеров для оценки
    """
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, labels)
        silhouette_scores.append(silhouette_avg)
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, max_k+1), silhouette_scores, marker='o')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Средний силуэтный коэффициент')
    plt.title('Силуэтный анализ для выбора количества кластеров')
    plt.show()

def cluster_embeddings(embeddings: List[List[float]], n_clusters: int) -> KMeans:
    """
    Кластеризует эмбеддинги с использованием KMeans.
    
    :param embeddings: Список эмбеддингов
    :param n_clusters: Количество кластеров
    :return: Обученная модель KMeans
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans  # Возвращаем обученную модель KMeans

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
        kmeans_model = pickle.load(f)
    return kmeans_model

