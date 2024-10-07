from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List
import pickle
from scipy.spatial.distance import cdist
import numpy as np


def find_optimal_clusters(embeddings: List[List[float]], max_k: int = 10) -> None:
    """
    Использует метод локтя и силуэтный анализ для определения оптимального количества кластеров.
    
    :param embeddings: Список эмбеддингов
    :param max_k: Максимальное количество кластеров для оценки
    """
    distortions = []
    inertias = []
    silhouette_scores = []
    
    K = range(2, max_k + 1)

    for k in K:
        # Строим и обучаем модель
        kmeanModel = KMeans(n_clusters=k, random_state=42).fit(embeddings)

        # Вычисляем distortion и inertia
        distortion = sum(np.min(cdist(embeddings, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / len(embeddings)
        inertia = kmeanModel.inertia_
        
        # Вычисляем силуэтный коэффициент
        labels = kmeanModel.predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, labels)
        
        distortions.append(distortion)
        inertias.append(inertia)
        silhouette_scores.append(silhouette_avg)

    # Находим максимальный силуэтный коэффициент и его соответствующее значение K
    max_silhouette_score = max(silhouette_scores)
    optimal_k = K[silhouette_scores.index(max_silhouette_score)]
    
    # Создаем график с тройными осями
    fig, ax1 = plt.subplots(figsize=(12, 6))  # Увеличиваем ширину графика

    # Первая ось для Distortion
    ax1.set_xlabel('Количество кластеров (K)')
    ax1.set_ylabel('Distortion', color='blue')
    ax1.plot(K, distortions, 'bx-', label='Distortion', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Вторая ось для Inertia
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Inertia', color='red')
    ax2.plot(K, inertias, 'rx-', label='Inertia', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Третья ось для Silhouette Score
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 70))  # Смещаем третью ось вправо
    ax3.set_ylabel('Silhouette Score', color='green')
    ax3.plot(K, silhouette_scores, 'gx-', label='Silhouette Score', color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    # Добавляем вертикальную и горизонтальную черные линии, ограниченные диапазоном осей
    ax3.axvline(x=optimal_k, ymin=0, ymax=(max_silhouette_score / max(silhouette_scores)), color='black', linestyle='--')
    

    # Добавляем аннотацию для максимального силуэтного коэффициента над точкой (с легким смещением вниз)
    ax3.text(optimal_k, max_silhouette_score - 0.03, 
             f'{max_silhouette_score:.2f}', 
             fontsize=10, verticalalignment='top', color='black')

    # Добавляем аннотацию количества кластеров под осью X, с легким смещением вниз
    ax1.text(optimal_k, ax1.get_ylim()[0] - 0.1 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), 
             f'K = {optimal_k}', 
             fontsize=10, verticalalignment='top', horizontalalignment='center', color='black')

    # Увеличиваем отступ сверху, чтобы заголовок не выходил за пределы
    plt.subplots_adjust(top=0.85, right=0.85)  # Увеличиваем отступы справа и сверху
    
    # Добавляем заголовок
    plt.title('Метод локтя и силуэтный анализ для выбора количества кластеров', pad=20)

    # Добавляем легенду и сетку
    ax1.grid(True)

    # Показываем график
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
