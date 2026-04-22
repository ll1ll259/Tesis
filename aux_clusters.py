import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


def ClusterMetrics(X, min_compo, max_compo, model=1, metric='distortion'):
    """
    Evalúa diferentes números de clusters para distintos modelos de clustering.

    Parámetros:
    -----------
    X : pandas.DataFrame o array
        Datos a utilizar para el clustering.

    min_compo : int
        Número mínimo de clusters a probar (usualmente >= 2).

    max_compo : int
        Número máximo de clusters a probar.

    model : int, default=1
        Modelo de clustering a utilizar:
        1 -> KMeans
        2 -> Clustering Jerárquico (Agglomerative)
        3 -> Gaussian Mixture Model (GMM)

    metric : str, default='distortion'
        Métrica para evaluar el número óptimo de clusters:
        - 'distortion' : suma de distancias cuadradas (KMeans/GMM)
        - 'silhouette' : coeficiente silhouette
        - 'calinski_harabasz' : índice Calinski-Harabasz
    """

    match model:

        case 1:
            km = KMeans(init='k-means++', random_state=1234)
            visualizer = KElbowVisualizer(
                km,
                k=(min_compo, max_compo),
                metric=metric,
                timings=False,
                force_model=True
            )
            visualizer.fit(X)
            visualizer.ax.set_title(f'{metric} Score Elbow for K-Means')
            visualizer.fig.set_size_inches(8, 6)
            return visualizer.show()

        case 2:
            aggl = AgglomerativeClustering(metric='euclidean')
            visualizer = KElbowVisualizer(
                aggl,
                k=(min_compo, max_compo),
                metric=metric,
                timings=False,
                force_model=True
            )
            visualizer.fit(X)
            visualizer.fig.set_size_inches(8, 6)
            visualizer.ax.set_title(f'{metric} Score Elbow for Agglomerative')
            return visualizer.show()

        case 3:
            salida = []

            for k in list(range(min_compo, max_compo)):
                gmm = GaussianMixture(
                    n_components=k,
                    init_params='k-means++',
                    random_state=1234
                )

                labels = gmm.fit_predict(X)

                if metric == 'calinski_harabasz':
                    score = calinski_harabasz_score(X, labels)

                elif metric == 'silhouette':
                    score = silhouette_score(X, labels, metric='euclidean')

                elif metric == 'distortion':
                    score = gmm.aic(X)

                salida.append(score)

            diferencias = [
                abs(salida[i + 1]) - abs(salida[i])
                for i in range(len(salida) - 1)
            ]

            posicion = min_compo + diferencias.index(max(diferencias)) #+ 1

            df_gmm = pd.DataFrame()
            df_gmm['N_clusters'] = range(min_compo, max_compo)
            df_gmm['score'] = salida

            plt.figure(figsize=(8, 6))
            plt.plot(list(range(min_compo, max_compo)), salida, marker='o')
            plt.axvline(
                x=posicion,
                linestyle='--',
                color='black',
                label=f'elbow at k={posicion}, score={round(salida[posicion - min_compo], 3)}'
            )
            plt.title(f'{metric} Score Elbow for Gaussian Mixture Model')
            plt.xlabel('$k$')
            plt.ylabel(f'{metric} score')
            plt.legend()
            plt.grid(True)
            plt.show()


def silueta(X, k):
    km = KMeans(n_clusters=k, init='k-means++', random_state=1234)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick',force_model=True)
    visualizer.fit(X)

    print(f"\nClusters : {k}")
    print(f"Score Silueta : {round(visualizer.silhouette_score_, 2)}")

    return visualizer.show()