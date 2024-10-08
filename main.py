import os
from src.preprocess import process_all_texts
from src.embeddings import generate_embeddings, save_embeddings, load_embeddings
from src.clustering import find_optimal_clusters, cluster_embeddings, save_model, load_model
from src.classify import find_max_similarity, find_nearest_neighbors
from src.visualization import visualize_clusters

def main():
    # Путь к директории с текстовыми файлами
    data_directory = "data/"
    
    # Путь к сохранённым эмбеддингам
    embeddings_file = "embeddings/embeddings.pkl"

    # Шаг 1: Генерация или загрузка эмбеддингов
    if os.path.exists(embeddings_file):
        print("Загрузка существующих эмбеддингов...")
        embeddings = load_embeddings(embeddings_file)
    else:
        print("Генерация эмбеддингов...")

        # Шаг 2: Чтение и предобработка текстов
        print("Чтение и предобработка текстов...")
        processed_texts = process_all_texts(data_directory)

        embeddings = generate_embeddings(processed_texts)
        save_embeddings(embeddings, embeddings_file)

    # Шаг 3: Кластеризация эмбеддингов
    print("Кластеризация эмбеддингов...")
    optimal_clusters = find_optimal_clusters(embeddings, max_k=30)  # Использует метод локтя и силуэтный коэффициент для определения оптимального количества кластеров.
    print(f"Рекомендуемое количество кластеров: {optimal_clusters}")
    
    n_clusters = int(input("Введите количество кластеров для KMeans: "))
    print(f"Кластеризация с использованием {n_clusters} кластеров")
    
    kmeans_model = cluster_embeddings(embeddings, n_clusters)
    cluster_labels = kmeans_model.labels_

    # Сохранение модели
    print("Сохранение модели кластеризации...")
    save_model(kmeans_model, "models/kmeans_model.pkl")

    # Шаг 4: Визуализация кластеров
    print("Визуализация кластеров...")
    visualize_clusters(embeddings, cluster_labels)  # Вызов функции для визуализации

    # Шаг 5: Классификация нового документа
    print("Классификация нового документа...")
    new_text = "This is a new document that needs to be classified."  # Пример нового текста
    new_embedding = generate_embeddings([new_text])[0]
    kmeans_model = load_model("models/kmeans_model.pkl")
    cluster_centroids = kmeans_model.cluster_centers_
    
    # Классификация нового документа с использованием метода косинусного расстояния.
    predicted_closest_cluster = find_max_similarity(new_embedding, cluster_centroids)

    # Классификация нового документа с использованием метода ближайших соседей.
    predicted_cluster_nbrs = find_nearest_neighbors(new_embedding, cluster_centroids)

    # Вывод результата классификации нового документа с использованием косинусного расстояния.
    print(f"Новый документ, при помощи метода косинусного расстояния, отнесен к кластеру: {predicted_closest_cluster}")

    # Вывод результата классификации нового документа с использованием метода ближайших соседей.
    print(f"Новый документ, при помощи метода ближайших соседей, отнесен к кластеру: {predicted_cluster_nbrs}")


if __name__ == "__main__":
    main()
