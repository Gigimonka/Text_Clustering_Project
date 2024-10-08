import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_clusters(embeddings, labels, perplexity=30):
    """
    Функция для визуализации кластеров с использованием метода t-SNE.
    
    :param embeddings: эмбеддинги данных (векторы)
    :param labels: метки кластеров для каждого эмбеддинга
    :param perplexity: параметр перплексии для t-SNE
    """
    
    print(f"Используется метод t-SNE для визуализации")
    reducer = TSNE(n_components=2)
    
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar()
    plt.title('Визуализация кластеров с использованием t-SNE')
    plt.xlabel("Компонента 1")
    plt.ylabel("Компонента 2")
    plt.show()
