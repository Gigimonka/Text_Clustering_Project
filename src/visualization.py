import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_elbow_and_silhouette(K, distortions, inertias, silhouette_scores, optimal_k, max_silhouette_score):
    """
    Визуализирует результаты метода локтя и силуэтного анализа для выбора оптимального количества кластеров.
    
    :param K: Диапазон значений кластеров
    :param distortions: Список искажений (distortion) для каждого значения K
    :param inertias: Список инерций для каждого значения K
    :param silhouette_scores: Список силуэтных коэффициентов для каждого значения K
    :param optimal_k: Оптимальное количество кластеров
    :param max_silhouette_score: Максимальный силуэтный коэффициент
    """
    # Построение графиков
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Distortion
    ax1.set_xlabel('Количество кластеров (K)')
    ax1.set_ylabel('Distortion', color='blue')
    ax1.plot(K, distortions, 'bx-', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Inertia
    ax2 = ax1.twinx()
    ax2.set_ylabel('Inertia', color='red')
    ax2.plot(K, inertias, 'rx-', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Silhouette Score
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 70))
    ax3.set_ylabel('Silhouette Score', color='green')
    ax3.plot(K, silhouette_scores, 'gx-', color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    # Вертикальная линия для оптимального K
    ax3.axvline(x=optimal_k, color='black', linestyle='--')

    # Аннотации
    ax3.text(optimal_k, max_silhouette_score - 0.03, f'{max_silhouette_score:.2f}', fontsize=10, verticalalignment='top', color='black')
    ax1.text(optimal_k, ax1.get_ylim()[0] - 0.1 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), f'K = {optimal_k}', fontsize=10, verticalalignment='top', horizontalalignment='center', color='black')

    # Настройки графика
    plt.subplots_adjust(top=0.85, right=0.85)
    plt.title('Метод локтя и силуэтный анализ для выбора количества кластеров', pad=20)
    ax1.grid(True)
    plt.show()
    

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
