<div class="cell markdown">

<h1><center>Workshop: K-means and PCA for segmentation</center></h1>

</div>

<div class="cell markdown">

# 1. Importation des librairies de Python nécessaires

</div>

<div class="cell code" execution_count="88">

``` python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
```

</div>

<div class="cell markdown">

# 2. Génération de données simulées

</div>

<div class="cell code" execution_count="89">

``` python
X, y = make_blobs(n_samples=500, n_features=5, centers=4, random_state=42)
```

</div>

<div class="cell markdown">

#### La taille des données

</div>

<div class="cell code" execution_count="90">

``` python
print(f"Taille des données: {X.shape}")
```

<div class="output stream stdout">

    Taille des données: (500, 5)

</div>

</div>

<div class="cell markdown">

# 3. Visualisation des Données

</div>

<div class="cell markdown">

les données ont plus de 4 dimensions, on doit réduire leur
dimensionnalité pour les visualiser.Pour cela on peut utiliser l'analyse
en composantes principales (**PCA**) pour réduire les données à 2
dimensions.

</div>

<div class="cell code" execution_count="104">

``` python
# Réduction de la dimensionnalité à 2 dimensions pour la visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualisation des données
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
plt.xlabel('Premier axe principal')
plt.ylabel('Deuxième axe principal')
plt.title('Visualisation des données après PCA')
plt.show()
```

<div class="output display_data">

![image](https://github.com/user-attachments/assets/c2798484-4aa2-4311-8928-b170c8d416b8)


</div>

</div>

<div class="cell markdown">

# 4. Implémentation de l’Algorithme K-means

</div>

<div class="cell markdown">

#### a. Initialisation Aléatoire

</div>

<div class="cell code" execution_count="105">

``` python
kmeans_random = KMeans(n_clusters=4, init='random', n_init=10, random_state=42)
kmeans_random.fit(X)

# Prédiction des clusters
labels_random = kmeans_random.labels_

# Affichage des centres
centers_random = kmeans_random.cluster_centers_
print("Centres des clusters (Initialisation Aléatoire):")
print(centers_random)
```

<div class="output stream stdout">

    Centres des clusters (Initialisation Aléatoire):
    [[-2.47555395  8.91844236  4.75402806  1.96802416 -6.95204041]
     [-6.98354008 -8.7518611   7.37901293  2.28804852  4.26234467]
     [-9.50789054  9.53000845  6.61491931 -5.63366645 -6.39047321]
     [-6.3345608  -3.93976884  0.39820355 -1.39284558 -3.97753497]]

</div>

</div>

<div class="cell markdown">

#### b. K-means++

</div>

<div class="cell code" execution_count="106">

``` python
kmeans_plus = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans_plus.fit(X)

# Prédiction des clusters
labels_plus = kmeans_plus.labels_

# Affichage des centres
centers_plus = kmeans_plus.cluster_centers_
print("Centres des clusters (K-means++):")
print(centers_plus)
```

<div class="output stream stdout">

    Centres des clusters (K-means++):
    [[-2.47555395  8.91844236  4.75402806  1.96802416 -6.95204041]
     [-6.98354008 -8.7518611   7.37901293  2.28804852  4.26234467]
     [-6.3345608  -3.93976884  0.39820355 -1.39284558 -3.97753497]
     [-9.50789054  9.53000845  6.61491931 -5.63366645 -6.39047321]]

</div>

</div>

<div class="cell markdown">

### Visualisation des Clusters

</div>

<div class="cell markdown">

#### Réduction des centres à 2 dimensions avec PCA

</div>

<div class="cell code" execution_count="94">

``` python
centers_random_pca = pca.transform(centers_random)
centers_plus_pca = pca.transform(centers_plus)
```

</div>

<div class="cell markdown">

#### Visualisation des clusters pour l'initialisation aléatoire

</div>

<div class="cell code" execution_count="107">

``` python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_random, palette='viridis')
plt.scatter(centers_random_pca[:, 0], centers_random_pca[:, 1], c='red', marker='x', s=200)
plt.xlabel('Premier axe principal')
plt.ylabel('Deuxième axe principal')
plt.title('Clusters avec Initialisation Aléatoire')
plt.show()
```

<div class="output display_data">

![image](https://github.com/user-attachments/assets/2ff0ec91-c9ce-4fa5-98b7-6ac6df1f8fed)


</div>

</div>

<div class="cell markdown">

#### Visualisation des clusters pour K-means++

</div>

<div class="cell code" execution_count="108">

``` python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_plus, palette='viridis')
plt.scatter(centers_plus_pca[:, 0], centers_plus_pca[:, 1], c='red', marker='x', s=200)
plt.xlabel('Premier axe principal')
plt.ylabel('Deuxième axe principal')
plt.title('Clusters avec K-means++')
plt.show()
```

<div class="output display_data">

![image](https://github.com/user-attachments/assets/c320acc9-f2ee-43a5-b671-03f389bcb332)

</div>

</div>

<div class="cell markdown">

# 5. Méthodes de Validation de Clustering

</div>

<div class="cell markdown">

#### Indice de Silhouette

</div>

<div class="cell code" execution_count="97">

``` python
silhouette_avg_random = silhouette_score(X, labels_random)
silhouette_avg_plus = silhouette_score(X, labels_plus)
print(f"Indice de silhouette (Initialisation Aléatoire): {silhouette_avg_random:.3f}")
print(f"Indice de silhouette (K-means++): {silhouette_avg_plus:.3f}")
```

<div class="output stream stdout">

    Indice de silhouette (Initialisation Aléatoire): 0.747
    Indice de silhouette (K-means++): 0.747

</div>

</div>

<div class="cell markdown">

#### Score de Calinski-Harabasz

</div>

<div class="cell code" execution_count="98">

``` python
calinski_harabasz_random = calinski_harabasz_score(X, labels_random)
calinski_harabasz_plus = calinski_harabasz_score(X, labels_plus)
print(f"Score de Calinski-Harabasz (Initialisation Aléatoire): {calinski_harabasz_random:.3f}")
print(f"Score de Calinski-Harabasz (K-means++): {calinski_harabasz_plus:.3f}")
```

<div class="output stream stdout">

    Score de Calinski-Harabasz (Initialisation Aléatoire): 3715.491
    Score de Calinski-Harabasz (K-means++): 3715.491

</div>

</div>

<div class="cell markdown">

#### Indice de Davies-Bouldin

</div>

<div class="cell code" execution_count="99">

``` python
davies_bouldin_random = davies_bouldin_score(X, labels_random)
davies_bouldin_plus = davies_bouldin_score(X, labels_plus)
print(f"Indice de Davies-Bouldin (Initialisation Aléatoire): {davies_bouldin_random:.3f}")
print(f"Indice de Davies-Bouldin (K-means++): {davies_bouldin_plus:.3f}")
```

<div class="output stream stdout">

    Indice de Davies-Bouldin (Initialisation Aléatoire): 0.364
    Indice de Davies-Bouldin (K-means++): 0.364

</div>

</div>

<div class="cell markdown">

#### Méthode de l'élbow pour déterminer le nombre optimal de clusters

</div>

<div class="cell code" execution_count="100">

``` python
def elbow_method(X, max_k):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Distorsion')
    plt.title("Méthode de l'élbow")
    plt.show()
```

</div>

<div class="cell code" execution_count="101">

``` python
elbow_method(X, max_k=10)
```

<div class="output display_data">

![image](https://github.com/user-attachments/assets/85ade38d-1780-466c-b378-c21572811842)


</div>

</div>

<div class="cell markdown">

# 6. Interprétation des Résultats

</div>

<div class="cell markdown">

### Indice de Silhouette :

</div>

<div class="cell markdown">

**Initialisation Aléatoire** : 0.747

**K-means++** : 0.747

**Interprétation**: Les deux méthodes d'initialisation (aléatoire et
K-means++) ont produit des clusters avec des indices de silhouette
identiques et élevés (0.747). les deux méthodes ont réussi à segmenter
les données de manière efficace et similaire.

</div>

<div class="cell markdown">

### Score de Calinski-Harabasz

</div>

<div class="cell markdown">

**Initialisation Aléatoire** : 3715.491

**K-means++** : 3715.491

**Interprétation**: Les scores de Calinski-Harabasz sont également
identiques pour les deux méthodes. Un score élevé de Calinski-Harabasz
indique une bonne densité et séparation des clusters. Les deux méthodes
d'initialisation ont donc produit des clusters de haute qualité.

</div>

<div class="cell markdown">

### Indice de Davies-Bouldin :

</div>

<div class="cell markdown">

**Initialisation Aléatoire** : 0.364

**K-means++** : 0.364

**Interprétation**: Les indices de Davies-Bouldin sont identiques pour
les deux méthodes. Un indice de Davies-Bouldin plus faible indique de
meilleurs clusters (moins de chevauchement entre clusters). Les deux
méthodes ont produit des clusters bien distincts et compacts.

</div>

<div class="cell markdown">

### Méthode de l'élbow

</div>

<div class="cell markdown">

-   Le graphe de la méthode de l'élbow montre une "coude" clair au
    niveau de 4 clusters.

-   Cela suggère que 4 est le nombre optimal de clusters pour ce jeu de
    données.

-   Au-delà de 4 clusters, la réduction de la distorsion devient
    marginale, ce qui signifie que l'ajout de clusters supplémentaires
    n'améliore pas significativement la segmentation.

</div>

<div class="cell markdown">

# 7. Meilleur modèle de Clustering

</div>

<div class="cell markdown">

Étant donné que tous les indicateurs de performance sont identiques pour
les deux méthodes d'initialisation, on peux pas conclure qu'une méthode
est meilleure que l'autre en termes de qualité des clusters.

Cependant, dans la pratique,**K-means++** est généralement préféré car
il tend à converger plus rapidement et de manière plus stable que
l'initialisation aléatoire. K-means++ choisit intelligemment les centres
initiaux pour accélérer la convergence et améliorer la qualité des
clusters.

</div>

<div class="cell markdown">

# 8. Peut on représenter les données avec les poids des centres obtenus ?

</div>

<div class="cell markdown">

Nous pouvons représenter les données en utilisant les coordonnées des
centres des clusters obtenus. Pour cela, nous allons effectuer une
réduction de dimensionnalité via l'Analyse en Composantes Principales
(ACP) et projeter les centres des clusters dans cet espace réduit.

</div>

<div class="cell markdown">

# 9. Analyse en Composantes Principales (PCA)

</div>

<div class="cell markdown">

#### a. Nouvelle Matrice des Observations

</div>

<div class="cell code" execution_count="102" scrolled="true">

``` python
from sklearn.decomposition import PCA

# Application de l'PCA pour réduire à 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_pca
```

<div class="output execute_result" execution_count="102">

    array([[-1.13447932e+01, -2.28643792e+00],
           [ 7.43938042e+00, -6.00222435e+00],
           [ 9.20866566e+00, -4.41724049e+00],
           [ 1.05715813e+01, -4.39376626e+00],
           [ 7.61055439e+00,  5.33125005e+00],
           [ 8.56835551e+00, -5.08576523e+00],
           [-4.20496288e+00,  2.07874122e+00],
           [-1.33014086e+01, -2.92341356e+00],
           [ 9.64302091e+00, -4.44946989e+00],
           [-4.28947862e+00,  1.18743711e+00],
           [ 7.49200832e+00,  5.52426104e+00],
           [-1.20916688e+01,  1.27208021e+00],
           [-1.22440633e+01, -1.49471669e+00],
           [-1.39047403e+01,  1.38195121e+00],
           [-1.29005389e+01, -1.85360451e-03],
           [ 7.56610579e+00, -2.91849003e+00],
           [ 7.64404327e+00,  4.44114034e+00],
           [-1.27987972e+01, -1.50719819e+00],
           [ 8.68238893e+00, -4.18133930e+00],
           [ 9.83515221e+00, -3.84074937e+00],
           [-3.02191158e+00,  2.87375477e-02],
           [-3.81937140e+00,  2.34246559e-01],
           [ 1.05067667e+01, -4.93999659e+00],
           [ 8.32825091e+00, -3.57862148e+00],
           [-2.94534320e+00,  8.95358332e-01],
           [ 6.48744800e+00,  4.22158546e+00],
           [-1.48755819e+01, -1.49937388e+00],
           [ 7.52271731e+00,  4.51937511e+00],
           [-3.51888802e+00,  1.72396897e+00],
           [-5.94124762e+00, -8.49400587e-02],
           [ 9.15690458e+00, -5.22966449e+00],
           [-1.33929055e+01, -4.34349764e-01],
           [ 9.69365739e+00, -4.40111989e+00],
           [ 8.97335935e+00, -3.49172234e+00],
           [-2.11023659e+00,  1.56906253e+00],
           [ 6.11412593e+00,  5.72322586e+00],
           [ 6.39798679e+00,  4.28208578e+00],
           [-1.47451420e+01, -3.06492570e+00],
           [ 6.55261355e+00,  4.52772822e+00],
           [-1.38374195e+01, -1.30271195e+00],
           [ 6.62110714e+00,  4.25560584e+00],
           [-5.22698504e+00, -3.99794928e-01],
           [-1.42108770e+01,  6.43672419e-01],
           [ 9.11481848e+00,  2.62658656e+00],
           [ 8.79816715e+00,  3.59405257e+00],
           [-1.32638221e+01, -8.75121297e-01],
           [-4.70325004e+00,  1.28815354e+00],
           [ 7.22984181e+00,  3.88735400e+00],
           [-5.25230688e+00,  9.41172698e-01],
           [-1.27627851e+01, -1.17758641e+00],
           [-1.31333237e+01, -1.62926020e+00],
           [-1.39985831e+01,  1.07110431e+00],
           [-2.66181266e+00,  1.52318174e+00],
           [ 8.44711854e+00,  4.72059015e+00],
           [ 9.30688435e+00, -4.04464093e+00],
           [ 6.78508107e+00,  4.46942019e+00],
           [ 8.81740621e+00,  5.51987407e+00],
           [-1.32650395e+01, -3.48565720e-01],
           [ 8.03052640e+00, -4.14841025e+00],
           [-3.71141450e+00,  1.01941640e-01],
           [ 7.98071978e+00,  3.93514126e+00],
           [ 7.61728808e+00, -6.10332129e+00],
           [-3.83119479e+00,  8.14351090e-01],
           [-1.21777472e+01, -7.54790431e-01],
           [-1.42377679e+01, -1.26838477e+00],
           [ 7.36150251e+00,  5.08297434e+00],
           [ 1.07064055e+01, -3.54240444e+00],
           [-1.19546965e+01,  5.13709383e-01],
           [ 7.04161831e+00,  6.01097357e+00],
           [-4.04667216e+00, -3.06476881e-01],
           [-6.10093483e+00,  5.40564570e-01],
           [-1.48259310e+00, -8.45266626e-01],
           [ 8.21618399e+00, -5.47278143e+00],
           [ 8.74138871e+00, -7.35613032e+00],
           [ 9.30441049e+00, -6.02825298e+00],
           [-5.38800570e+00,  1.90183947e+00],
           [ 9.43398895e+00, -4.48901621e+00],
           [-1.31020684e+01,  5.72314159e-01],
           [ 9.34614951e+00, -6.16876533e+00],
           [-1.14363373e+01,  7.30385624e-01],
           [-3.86582879e+00,  9.12107097e-01],
           [-1.34956170e+01, -2.37951243e+00],
           [ 9.79459518e+00, -4.90209359e+00],
           [-3.31525160e+00,  1.78864659e-01],
           [-3.47538989e+00,  5.92553036e-01],
           [-4.11895179e+00, -6.94562305e-01],
           [ 6.09285028e+00,  6.09198639e+00],
           [ 9.09186887e+00, -5.37670186e+00],
           [ 7.85224868e+00,  6.21939655e+00],
           [-3.57230742e+00,  1.10054152e-01],
           [ 8.28532062e+00, -5.45794107e+00],
           [ 8.01351855e+00,  3.83052123e+00],
           [ 8.05502330e+00,  5.21172722e+00],
           [-1.24189669e+01, -1.54680694e+00],
           [-1.27107180e+01, -1.13199286e+00],
           [ 8.26848819e+00,  5.25014106e+00],
           [ 7.37486851e+00,  5.16974034e+00],
           [ 1.02899770e+01, -3.62848266e+00],
           [ 9.36057034e+00, -5.11907987e+00],
           [-1.38555410e+01,  3.51858755e-01],
           [-1.27967895e+01, -2.62193602e+00],
           [ 1.07716107e+01, -4.94327477e+00],
           [ 6.97023447e+00,  4.08373805e+00],
           [ 1.09010435e+01, -5.98754386e+00],
           [-1.30681763e+01,  4.16602115e-01],
           [-1.50829899e+01, -9.59201250e-01],
           [-1.10838264e+01, -2.14394736e+00],
           [-1.23208027e+01,  2.98186117e-01],
           [ 8.33709446e+00,  7.10611729e+00],
           [-4.66008703e+00,  4.84835247e-01],
           [ 9.85416959e+00, -4.71565438e+00],
           [-1.39943397e+01, -2.02815092e+00],
           [-2.96741059e+00,  2.02434297e+00],
           [-5.15605159e+00,  2.61458705e+00],
           [ 9.96954239e+00, -5.63924572e+00],
           [ 8.19233309e+00, -4.11160111e+00],
           [-3.56434485e+00,  9.30714186e-01],
           [-1.26586631e+01, -7.96431504e-01],
           [ 9.78578608e+00, -7.73940191e+00],
           [-1.36884311e+01, -4.16199912e-01],
           [-5.53508016e+00,  1.67349586e+00],
           [-1.52941429e+01,  1.42640670e+00],
           [ 8.02815430e+00,  6.35661562e+00],
           [-4.58906887e+00, -7.45015235e-01],
           [ 8.30063892e+00,  7.53799326e+00],
           [ 9.29092017e+00, -5.54169034e+00],
           [ 9.54423031e+00, -5.97686724e+00],
           [ 8.03948585e+00,  4.34577584e+00],
           [-9.68269230e+00, -1.22290329e+00],
           [ 6.27077032e+00,  4.17013269e+00],
           [-1.33721187e+01, -6.28139691e-01],
           [-4.73740034e+00,  1.91337650e+00],
           [-1.13027050e+01, -1.23456903e-02],
           [ 6.34941876e+00,  5.28208612e+00],
           [-5.21089980e+00,  8.73871501e-02],
           [-4.72115672e+00,  5.77985319e-02],
           [ 8.57721748e+00,  4.72642935e+00],
           [ 7.34190543e+00,  4.66964910e+00],
           [ 9.52789916e+00,  5.22879039e+00],
           [-1.24721042e+01, -3.40335788e-01],
           [ 1.00367606e+01, -4.68600614e+00],
           [ 5.65127953e+00,  5.40810178e+00],
           [ 7.98231422e+00,  4.01821441e+00],
           [ 5.97321135e+00,  6.28205330e+00],
           [-1.18977945e+01, -9.76932971e-01],
           [ 1.01632893e+01, -4.11775614e+00],
           [-1.42195714e+01, -9.91718009e-01],
           [-1.19827602e+01, -8.17453004e-01],
           [-3.58898123e+00,  8.24401013e-01],
           [-3.81609457e+00, -5.81786500e-01],
           [-4.92907038e+00, -9.53607624e-01],
           [-4.32844165e+00,  2.96919875e+00],
           [-4.26599222e+00,  2.90791399e-01],
           [-1.35354925e+01, -3.17308672e+00],
           [-2.76201186e+00,  2.53041097e+00],
           [-3.98907979e+00, -3.33331106e-01],
           [ 8.61688976e+00, -5.00485280e+00],
           [-1.39891049e+01, -1.90322542e+00],
           [ 9.10710048e+00,  6.48528606e+00],
           [ 1.07692353e+01, -4.36501432e+00],
           [ 9.29950526e+00,  3.38165507e+00],
           [ 9.14084346e+00, -4.39804702e+00],
           [-1.29765744e+01,  7.59209137e-01],
           [-5.40070068e+00,  1.94377541e+00],
           [ 8.85669632e+00, -6.30238535e+00],
           [ 7.11980234e+00,  4.21883762e+00],
           [-4.79348902e+00,  2.71061289e-01],
           [-4.01322257e+00,  5.70065144e-01],
           [ 8.26525398e+00,  4.50745316e+00],
           [-5.03471719e+00,  1.08573987e+00],
           [ 9.24799012e+00, -5.31086731e+00],
           [ 1.10205781e+01, -4.60434817e+00],
           [-5.45174492e+00, -1.38921529e+00],
           [-1.27174153e+01,  6.20954851e-01],
           [ 1.02340357e+01, -6.48683957e+00],
           [-1.25457520e+01, -1.24775111e+00],
           [ 8.80189703e+00, -6.39411941e+00],
           [-3.79712872e+00,  6.83488810e-01],
           [ 8.16170903e+00,  6.53411976e+00],
           [-1.26733154e+01,  4.18826179e-01],
           [-4.01792876e+00,  6.08871758e-01],
           [-1.24816268e+01, -1.17050254e+00],
           [ 6.75442153e+00,  4.95061773e+00],
           [ 1.09056817e+01, -6.64210556e+00],
           [-1.43993300e+01, -1.18872368e+00],
           [-5.03497429e+00,  4.31418167e-01],
           [ 9.37063195e+00, -3.92235193e+00],
           [ 8.76222325e+00,  5.49978803e+00],
           [ 1.00743481e+01,  5.58950937e+00],
           [-1.27995630e+01, -4.81803445e-01],
           [ 7.12981333e+00,  6.02629452e+00],
           [-1.38059197e+01, -1.17133107e+00],
           [ 9.51783616e+00, -4.32690045e+00],
           [ 6.51132613e+00,  4.40428706e+00],
           [ 8.69894997e+00, -5.72764231e+00],
           [ 9.50439764e+00,  4.64507797e+00],
           [-3.47056635e+00, -1.36400550e+00],
           [ 8.73526337e+00, -5.79041263e+00],
           [ 1.00227111e+01,  5.07947358e+00],
           [ 9.88733933e+00, -6.50457850e+00],
           [ 8.68324528e+00, -5.94669128e+00],
           [ 8.93796141e+00, -3.73831498e+00],
           [ 9.49261747e+00,  5.26839734e+00],
           [ 1.04015088e+01, -4.27457218e+00],
           [ 9.68368468e+00, -4.00914614e+00],
           [-1.26773207e+01, -9.24118831e-01],
           [-1.34108838e+01, -1.62229353e+00],
           [ 8.87122711e+00, -2.94058939e+00],
           [ 7.59094503e+00,  5.04596020e+00],
           [-4.41161519e+00,  1.45010743e+00],
           [-1.24345034e+01,  1.22789840e-01],
           [ 1.00812723e+01, -4.69281672e+00],
           [-4.52506977e+00,  2.91233431e-01],
           [ 8.69283053e+00, -4.31330256e+00],
           [ 9.34269524e+00, -4.73625798e+00],
           [ 5.77807394e+00,  5.68462074e+00],
           [ 7.28632442e+00,  4.73097031e+00],
           [-1.25791983e+01,  4.13404043e-01],
           [ 9.10021852e+00, -6.32918111e+00],
           [ 7.96499865e+00, -4.04342080e+00],
           [ 8.32761127e+00,  5.33946199e+00],
           [ 9.35931116e+00, -5.31275733e+00],
           [ 9.59329252e+00, -3.78208067e+00],
           [ 8.47378974e+00,  6.33903823e+00],
           [-5.31552038e+00, -3.62824408e-01],
           [ 7.68460932e+00,  3.63117504e+00],
           [ 1.07167605e+01, -4.60586409e+00],
           [-3.92726276e+00, -9.56868836e-01],
           [-1.20096556e+01,  2.10053656e-01],
           [ 7.85764873e+00, -2.74585288e+00],
           [ 1.10554441e+01, -5.39350485e+00],
           [ 7.65155229e+00,  6.32157012e+00],
           [ 7.26471709e+00,  4.68264486e+00],
           [-1.22983463e+01, -2.88687817e-01],
           [ 7.86695903e+00,  6.72398481e+00],
           [ 6.17075082e+00,  6.04084076e+00],
           [ 8.14293994e+00,  4.46539216e+00],
           [ 1.05579783e+01, -5.91632871e+00],
           [ 1.03862429e+01, -8.89716012e+00],
           [-4.88359744e+00,  6.19975584e-01],
           [ 9.15405376e+00, -3.81698204e+00],
           [ 9.90594019e+00, -5.82627059e+00],
           [-1.28050187e+01,  4.73803987e-01],
           [ 6.06259549e+00,  4.38455669e+00],
           [-4.46531096e+00,  2.10231448e-01],
           [-1.30768184e+01, -7.78274337e-01],
           [ 7.40309903e+00,  4.94067117e+00],
           [-1.23049693e+01, -1.36242579e+00],
           [ 8.06599778e+00,  3.99441043e+00],
           [ 8.63869730e+00,  7.65328654e+00],
           [-5.42064043e+00,  9.69461728e-01],
           [-1.25286910e+01,  3.43300367e-01],
           [-2.74325193e+00,  1.40106591e+00],
           [ 9.17151287e+00, -4.69242313e+00],
           [ 1.04427235e+01, -5.01941766e+00],
           [ 9.11109624e+00, -5.22276831e+00],
           [-1.25632003e+01, -8.22603671e-01],
           [ 7.58941911e+00,  5.52910305e+00],
           [ 9.42360204e+00, -4.44136604e+00],
           [-1.40443444e+01, -4.26768574e-01],
           [-2.99800720e+00,  7.96957912e-01],
           [ 9.21021281e+00, -5.61223006e+00],
           [-3.64243331e+00,  1.25300038e+00],
           [ 9.31050947e+00, -6.35456075e+00],
           [ 9.02934579e+00, -5.14768676e+00],
           [-1.28670307e+01, -1.24877288e+00],
           [ 7.95084881e+00,  5.09136318e+00],
           [-1.08243245e+01, -1.72177874e+00],
           [ 7.05278429e+00,  5.72445813e+00],
           [-3.32817640e+00,  8.30693865e-01],
           [-3.22132048e+00,  1.90737188e+00],
           [ 9.49872246e+00, -6.39941817e+00],
           [-1.20258177e+01,  1.61979127e+00],
           [-3.25652511e+00,  4.82265299e-02],
           [-1.26588263e+01,  6.22603394e-02],
           [ 1.07841666e+01, -5.35166865e+00],
           [ 1.02062789e+01, -5.38295494e+00],
           [-4.13447963e+00,  8.34035363e-01],
           [ 6.64850808e+00,  5.35999117e+00],
           [ 8.01576742e+00,  5.95724109e+00],
           [ 9.75283018e+00,  7.82941886e+00],
           [ 1.04970420e+01, -6.59716241e+00],
           [-2.92198524e+00,  2.02738151e+00],
           [-1.12283881e+01, -2.55526596e+00],
           [ 1.04109637e+01, -3.12019611e+00],
           [-2.49303323e+00, -8.79800128e-01],
           [ 7.78493869e+00,  5.50614963e+00],
           [ 8.91260416e+00, -6.80590566e+00],
           [-1.25303020e+01, -1.26198809e-01],
           [ 8.37959957e+00, -6.05692517e+00],
           [ 8.79942158e+00, -5.52760891e+00],
           [ 6.30082766e+00,  4.44571587e+00],
           [-5.12918865e+00, -1.78195793e-02],
           [ 9.18289986e+00, -4.34108332e+00],
           [ 9.28199653e+00, -5.80119446e+00],
           [ 7.21781654e+00,  6.90700205e+00],
           [-1.11777186e+01, -1.03320365e+00],
           [-1.19046614e+01, -1.25812950e+00],
           [-5.14807177e+00,  1.12650168e+00],
           [-3.53408073e+00,  6.89863450e-02],
           [-1.17516848e+01, -1.28843705e-01],
           [-3.79132121e+00,  3.96233944e-01],
           [ 7.97066693e+00, -5.11766177e+00],
           [ 7.90157320e+00, -4.27004343e+00],
           [-4.33670228e+00,  6.23362749e-01],
           [ 8.70452298e+00,  5.99783257e+00],
           [-1.29859954e+01, -1.53142783e+00],
           [ 8.07342037e+00, -6.40210480e+00],
           [ 9.28563683e+00, -3.74007202e+00],
           [-4.90103296e+00,  8.86700532e-01],
           [-4.50786839e+00,  2.80683166e+00],
           [ 7.43589313e+00,  7.41783387e+00],
           [ 7.77142165e+00,  5.13436949e+00],
           [-3.39787316e+00,  8.00685722e-01],
           [ 8.36371629e+00,  6.69958058e+00],
           [-1.25767793e+01, -2.91531723e+00],
           [-4.39437347e+00,  1.12707166e+00],
           [ 7.52014808e+00,  5.87022549e+00],
           [ 8.43339553e+00,  5.56384580e+00],
           [-6.29264101e+00,  6.74434280e-01],
           [ 9.61500572e+00, -5.91448092e+00],
           [-1.22706251e+01, -2.41523082e-01],
           [ 8.47911138e+00,  4.55213115e+00],
           [ 9.91011172e+00, -5.07329894e+00],
           [ 9.43570578e+00, -4.69532972e+00],
           [-4.59902744e+00, -1.72556356e+00],
           [ 8.29859039e+00,  5.58485303e+00],
           [-1.35147437e+01,  1.31807117e+00],
           [ 7.82344856e+00,  4.44320013e+00],
           [-4.29693538e+00,  1.52041481e+00],
           [ 9.10414519e+00, -4.98555515e+00],
           [ 6.95174531e+00,  6.40383290e+00],
           [-1.40286901e+01, -7.38766506e-01],
           [ 7.56485246e+00,  6.65575040e+00],
           [-4.91364829e+00,  7.52969281e-01],
           [ 8.30848370e+00,  5.64927573e+00],
           [-1.22625356e+01, -6.86940981e-01],
           [ 9.12878824e+00, -5.93750578e+00],
           [ 8.89960030e+00, -6.34134407e+00],
           [-1.15540039e+01, -1.67516104e+00],
           [-3.93257591e+00, -2.10155784e-01],
           [-1.31611792e+01, -2.64300608e+00],
           [-4.06830782e+00,  3.95893108e-01],
           [ 7.93283232e+00,  5.08248160e+00],
           [-1.37036161e+01,  2.61396733e-01],
           [-1.18132812e+01, -8.74923573e-01],
           [-1.31307549e+01, -3.23076880e-01],
           [-3.29485048e+00,  1.35186282e+00],
           [-1.31733636e+01, -1.20809759e+00],
           [-1.24688720e+01, -1.02711947e+00],
           [ 8.42103344e+00,  6.23472050e+00],
           [ 8.51710732e+00,  6.62117665e+00],
           [-1.31848615e+01, -2.94053905e-01],
           [-1.17633935e+01,  4.26791474e-02],
           [ 8.61261929e+00, -5.78925787e+00],
           [-4.06404463e+00,  1.68361980e-01],
           [ 1.05262914e+01, -5.56759732e+00],
           [-5.79746867e+00,  2.20517813e-01],
           [-1.27557785e+01, -9.92036469e-01],
           [-1.28524890e+01, -2.44679048e+00],
           [-3.22566107e+00,  3.47977568e-01],
           [-3.83534292e+00, -9.48497285e-01],
           [ 5.99172031e+00,  4.20920520e+00],
           [-4.97203631e+00, -4.64793908e-01],
           [ 8.27661058e+00, -4.25277762e+00],
           [ 9.32946910e+00, -5.46523747e+00],
           [ 7.58721597e+00,  7.13610340e+00],
           [ 8.58919160e+00,  4.90183784e+00],
           [-1.28849325e+01, -1.20156970e+00],
           [ 7.05862128e+00, -5.56216513e+00],
           [-5.99338368e+00, -3.85588780e-01],
           [-1.29210862e+01, -3.84747359e-01],
           [-4.76012107e+00,  1.17584918e+00],
           [ 6.58183468e+00,  5.28706706e+00],
           [ 8.64126361e+00,  7.44612280e+00],
           [-1.41415471e+01,  2.61033614e-01],
           [ 9.58875505e+00, -3.97988412e+00],
           [ 7.66672024e+00, -4.55982493e+00],
           [ 7.68340014e+00,  3.65327579e+00],
           [-1.28167005e+01,  6.33640543e-01],
           [-4.12397337e+00,  1.43351146e+00],
           [ 7.91810088e+00,  4.63376235e+00],
           [-1.24623515e+01,  6.17635444e-01],
           [-1.54622068e+01, -3.86865490e-01],
           [ 8.58248111e+00, -5.25292138e+00],
           [ 7.73018129e+00,  3.38515940e+00],
           [-3.82018185e+00,  1.00897032e+00],
           [-4.71208084e+00,  3.38559901e-01],
           [ 8.94110232e+00,  3.66643870e+00],
           [ 9.26759815e+00, -6.57847816e+00],
           [-3.63857772e+00,  1.03535940e+00],
           [ 9.72583055e+00, -7.16256181e+00],
           [ 9.47598210e+00, -5.14372358e+00],
           [ 8.00673869e+00,  4.86907168e+00],
           [ 8.78576325e+00,  5.54744742e+00],
           [ 9.98572470e+00, -4.73548753e+00],
           [ 9.75278514e+00, -4.64647379e+00],
           [-1.35406732e+01,  1.96886500e+00],
           [-4.72401383e+00, -1.22144909e+00],
           [-4.57981605e+00, -4.02038148e-01],
           [ 8.47985248e+00, -6.94390155e+00],
           [-1.33884676e+01,  1.34704423e+00],
           [ 1.02353195e+01,  4.75856564e+00],
           [-1.51018200e+01, -1.29342236e+00],
           [-3.88154244e+00,  2.16077304e+00],
           [ 8.01931844e+00,  5.65297340e+00],
           [-3.88115508e+00,  5.31469684e-01],
           [-4.52655844e+00, -1.55358800e+00],
           [-3.61730318e+00, -1.35738866e+00],
           [ 8.53422762e+00, -5.29324531e+00],
           [-4.00170590e+00, -4.15290665e-01],
           [ 7.10825725e+00, -4.41432554e+00],
           [-1.21310932e+01,  3.09947769e-01],
           [ 8.95747907e+00,  6.74816093e+00],
           [-5.43495660e+00,  6.35398620e-01],
           [-1.37044650e+01,  6.14738361e-02],
           [-3.77943746e+00,  9.20799182e-01],
           [ 1.00542463e+01, -6.05734085e+00],
           [ 9.09851308e+00,  6.97403171e+00],
           [-1.16865297e+01, -7.10461707e-01],
           [-3.77245045e+00,  1.03376243e+00],
           [-5.01061567e+00,  2.55321193e+00],
           [-4.28300184e+00,  1.30203625e+00],
           [-4.37140062e+00,  3.87838556e-01],
           [-1.28083813e+01, -1.31216207e+00],
           [-1.39310182e+01, -2.38853656e+00],
           [-5.30035395e+00,  6.83777525e-01],
           [ 7.79337687e+00,  4.94914381e+00],
           [-6.08404196e+00, -1.32093911e+00],
           [-1.03123831e+01, -5.89459359e-01],
           [-1.22820006e+01, -1.30418119e+00],
           [ 8.68389147e+00,  4.66542612e+00],
           [ 7.55275300e+00,  5.72927083e+00],
           [-1.22572454e+01, -2.22448421e-01],
           [-1.35910999e+01, -9.57835383e-01],
           [-1.13832842e+01, -1.90290869e-01],
           [-4.01814270e+00,  1.33017335e+00],
           [ 7.23659499e+00,  4.12693152e+00],
           [-1.34190771e+01, -8.75981985e-01],
           [ 9.45977924e+00,  6.45581369e+00],
           [-4.06648825e+00, -1.69700936e-01],
           [-3.75603683e+00,  1.82632994e+00],
           [-1.29233134e+01, -2.25746696e+00],
           [-1.28527898e+01,  6.05572345e-01],
           [-1.20970379e+01, -1.26911827e+00],
           [-3.52869136e+00,  1.26650950e-01],
           [ 9.48735016e+00, -4.61815303e+00],
           [-1.26713238e+01, -9.11977494e-01],
           [-3.93938090e+00, -7.08989712e-01],
           [ 7.22775303e+00,  4.23671184e+00],
           [ 7.38850628e+00,  4.99527762e+00],
           [ 8.99853538e+00,  5.42663619e+00],
           [-1.34205853e+01, -2.62161663e+00],
           [-3.73517804e+00,  1.45139330e+00],
           [ 6.52711677e+00,  6.65144811e+00],
           [ 7.84534439e+00,  3.99799591e+00],
           [-1.33674244e+01, -1.93724091e+00],
           [-5.68863696e+00, -6.44204074e-01],
           [ 8.16850582e+00,  4.76584139e+00],
           [-4.55941200e+00,  6.55321461e-01],
           [ 9.04987338e+00,  7.38530045e+00],
           [ 8.03856593e+00, -4.99485523e+00],
           [ 8.67309653e+00,  4.23448498e+00],
           [-1.10938507e+01, -1.41870918e-01],
           [-1.28325136e+01, -6.92748110e-01],
           [-5.39249449e+00,  2.47745263e+00],
           [-1.35902759e+01, -1.47164621e-01],
           [-1.31874830e+01,  4.54595487e-01],
           [ 1.01418712e+01, -5.14316825e+00],
           [-3.11599838e+00, -4.08431724e-01],
           [-1.33759566e+01, -1.36907246e+00],
           [-4.24800582e+00,  1.13238712e-01],
           [ 1.02059350e+01, -6.73117312e+00],
           [-4.73581566e+00,  1.37794877e+00],
           [-4.06142456e+00, -5.73988440e-01],
           [-5.89276667e+00,  1.09552714e+00],
           [-5.80315923e+00,  1.31886852e+00],
           [ 6.73619459e+00,  5.95268594e+00],
           [-4.28517514e+00,  3.86519759e-01],
           [ 7.27101755e+00,  4.91110077e+00],
           [-4.63112596e+00,  1.87991864e-01],
           [ 8.89612704e+00, -5.99971027e+00],
           [ 7.66234386e+00,  4.29104257e+00],
           [-2.60859414e+00,  3.17877933e-02],
           [-1.04928160e+01,  3.00993003e-01],
           [ 7.35716067e+00,  7.98317012e+00],
           [ 8.28474144e+00,  4.85593204e+00],
           [ 7.41617222e+00,  4.48799687e+00],
           [ 8.36140874e+00,  4.58066293e+00],
           [-1.36324402e+01, -8.72903168e-01],
           [ 1.12091181e+01, -5.25985641e+00],
           [-1.32340948e+01, -1.60306741e+00],
           [ 9.02332078e+00, -4.22944201e+00],
           [ 7.48743477e+00,  3.69772290e+00],
           [ 9.01016923e+00, -5.47276622e+00],
           [ 7.62916897e+00, -4.98607355e+00],
           [ 7.93936801e+00, -4.38685464e+00],
           [ 8.75552454e+00, -5.03818389e+00],
           [-1.27221961e+01, -1.80084135e+00],
           [-4.27659160e+00,  1.87393509e+00]])

</div>

</div>

<div class="cell markdown">

#### b. Valeurs Propres et Vecteurs Propres Associés aux Axes Principaux

</div>

<div class="cell code" execution_count="103">

``` python
eigenvalues = pca.explained_variance_
print("Valeurs propres :")
print(eigenvalues)

eigenvectors = pca.components_
print("Vecteurs propres :")
print(eigenvectors)
```

<div class="output stream stdout">

    Valeurs propres :
    [83.73533925 14.84968182]
    Vecteurs propres :
    [[ 0.0275173   0.86927694  0.00647878 -0.18277972 -0.45842121]
     [ 0.67774994  0.03427744 -0.25389588  0.66923612 -0.16474214]]

</div>

</div>

<div class="cell markdown">

#### c. Inertie de Chaque Axe

</div>

<div class="cell code" execution_count="77">

``` python
explained_variance_ratio = pca.explained_variance_ratio_
print("Inertie de chaque axe :")
print(explained_variance_ratio)
```

<div class="output stream stdout">

    Inertie de chaque axe :
    [0.7437459 0.1318964]

</div>

</div>

<div class="cell markdown">

#### d. Somme des Inerties

</div>

<div class="cell code" execution_count="81">

``` python
# Vérification de la somme des inerties
sum_explained_variance_ratio = np.sum(explained_variance_ratio)
print(f"Somme des inerties : {sum_explained_variance_ratio:.3f}")
```

<div class="output stream stdout">

    Somme des inerties : 0.876

</div>

</div>

<div class="cell markdown">

#### e. Représentation des Données et Centres sur les Deux Axes Principaux

</div>

<div class="cell code" execution_count="85">

``` python
# Implémentation de K-means avec initialisation K-means++
kmeans_plus = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans_plus.fit(X)

# Prédiction des clusters
labels_plus = kmeans_plus.labels_

# Affichage des centres
centers_plus = kmeans_plus.cluster_centers_

# Projection des centres dans l'espace réduit pour visualisation
centers_plus_pca = pca.transform(centers_plus)

# Visualisation des clusters avec K-means++
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_plus, palette='viridis')
plt.scatter(centers_plus_pca[:, 0], centers_plus_pca[:, 1], c='red', marker='x', s=200)
plt.xlabel('Premier axe principal')
plt.ylabel('Deuxième axe principal')
plt.title('Clusters avec K-means++')
plt.show()
```

<div class="output display_data">

![image](https://github.com/user-attachments/assets/27accef0-4e60-48fe-9fa2-d8652676d931)


</div>

</div>

<div class="cell markdown">

#### f. Interprétation des Résultats

</div>

<div class="cell markdown">

**Valeurs Propres et Vecteurs Propres** :

Les valeurs propres indiquent la quantité de variance expliquée par
chaque axe principal. Les vecteurs propres (ou axes principaux) montrent
la direction des axes principaux dans l'espace des caractéristiques
d'origine.

**Inertie de Chaque Axe**:

L'inertie de chaque axe représente la proportion de la variance totale
expliquée par cet axe. Une inertie élevée sur les premiers axes est
souhaitable, car cela signifie que ces axes capturent la majorité de la
variance des données.

**Représentation des Données et des Centres** :

La visualisation montre la distribution des données et la position des
centres des clusters dans les deux dimensions principales. Les centres
des clusters, représentés par des croix rouges, indiquent les points
moyens autour desquels les données sont groupées. Une bonne séparation
des clusters dans cet espace réduit indique que les clusters sont bien
formés.

</div>
