from src.preprocess import process_all_texts
from src.embeddings import generate_embeddings, save_embeddings, load_embeddings
from src.clustering import find_optimal_clusters, cluster_embeddings, save_model, load_model
from src.classify import classify_new_document

def main():
    # Путь к директории с текстовыми файлами
    data_directory = "data/"

    # Шаг 1: Чтение и предобработка текстов
    print("Чтение и предобработка текстов...")
    processed_texts = process_all_texts(data_directory)

    # Шаг 2: Генерация эмбеддингов
    print("Генерация эмбеддингов...")
    embeddings = generate_embeddings(processed_texts)
    save_embeddings(embeddings, "embeddings/embeddings.pkl")

    # Шаг 3: Кластеризация эмбеддингов
    print("Кластеризация эмбеддингов...")
    find_optimal_clusters(embeddings, max_k=10)  # Можно использовать для определения оптимального K
    n_clusters = 5  # Например, выбрали 5 кластеров
    kmeans_model = cluster_embeddings(embeddings, n_clusters)

    # Сохранение модели
    print("Сохранение модели кластеризации...")
    save_model(kmeans_model, "models/kmeans_model.pkl")

    # Шаг 4: Классификация нового документа
    print("Классификация нового документа...")
    new_text = "This is a new document that needs to be classified."  # Пример нового текста
    new_embedding = generate_embeddings([new_text])[0]
    kmeans_model = load_model("models/kmeans_model.pkl")
    cluster_centroids = kmeans_model.cluster_centers_  
    predicted_cluster = classify_new_document(new_embedding, cluster_centroids)

    print(f"Новый документ отнесен к кластеру: {predicted_cluster}")

if __name__ == "__main__":
    main()
