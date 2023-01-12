## Implémentation d'un CNN - LeNet-5 sur GPU

## 1. Introduction

### Quel IDE pour compiler le projet ?
Peu d'IDE propose une gestion de CUDA propre mais une coloration synthaxique du langage C conviendra très bien.
Vous êtes libre dans le choix de l'IDE pour développer en CUDA mais la compilation et l'éxécution se feront via la console. 
Un simple éditeur de texte fera donc l'affaire. VsCode, cLion ou encore Jupyter-Lab seront ainsi des options envisageables.
Le langage utilisé pour ce projet étant du CUDA, un PC avec une carte graphique Nvidia sera nécessaire, au risque de ne pas pouvoir utiliser les fonctions optimisées pour le GPU. 


### Compilation et Execution depuis la console
La compilation du Cuda se fait via la commande suivante : ```nvcc nom_fichier.cu -o nom_fichier```

Une fois cette commande rentrée, il ne reste plus qu'à exécuter le fichier créé via la commande suivante : ```./nom_fichier```




## 2. Objectifs
Les objectifs de ce projet de Hardware for Signal Processing sont : 
- Apprendre à utiliser CUDA
- Etudier la complexité d'algorithmes et l'accélération obtenue sur GPU par rapport à une exécution sur CPU
- Observer les limites de l'utilisation d'un GPU
- Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement
- Exporter des données depuis un notebook python et les réimporter dans un projet cuda
- Faire un suivi du  projet et du versionning grâce à l'outil git




## 3. Partie 1 - Prise en main de Cuda : Addition et Multiplication de matrices
Cette partie 1 correspond au fichier TP_Intro.

### 3.1. Initialisation de matrices
On commence, dans la fonction d'initialisation, par initialiser la matrice avec des valeurs aléatoires qu'on insère dans une liste.

Dans le cas de l'utilisation de matrices sur GPU, deux opérations supplémentaires sont nécessaires :
- L'allocation des mémoires des matrices sur le GPU à l'aide de la fonction : ```cudaMalloc``` 
- Leur copie depuis le CPU vers le GPU avec ```cudaMemcpy```

Pour permettre la parallélisation, il est nécessaire de définir les dimensions des variables _dim3_ (dont il sera question dans la partie suivante) avec "n" et "m" les dimensions de la matrice :

- ```dim3 block_size(n, m);```

- ```dim3 grid_size(1, 1);```


### 3.2. Addition de matrices
#### **Sur CPU**
Sur CPU, on additionne deux matrices (représentées sous forme de listes) en procèdant de manière classique, à savoir en additionnant les coefficients de chacune deux à deux.

#### **Sur GPU**
Sur GPU, le procédé reste le même mais la parallélisation des calculs le rend plus complexe à mettre en oeuvre .
La principale difficulté réside dans la définition des indices des coefficients matriciels.
Dans ce cas, celle-ci se fait cette fois via des variables permettant de définir les threads afin de paralléliser les calculs.

L'indice de la colonne à considérer sera ainsi définie par plusieurs paramètres :
- blockIdx.x qui désigne le numéro de la colonne du "block" dans la "grid"
- blockDim.x qui désigne le nombre total de colonnes de ce "block"
- threadIdx.x qui désigne le numéro de la colonne du "thread" appartenant au "block"

De plus, il est nécessaire de définir, pour la parallélisation, la fonction avec l'indice "global" afin d’effectuer les calculs sur le GPU en l’appelant depuis le CPU.


### 3.3. Multiplication de matrices
#### **Sur CPU**
La multiplication de deux matrices sur le CPU se fait, comme pour l'addition sur CPU, de façon habituelle.

#### **Sur GPU**
Comme pour précédemment, la multiplication sur GPU repose sur le même principe que la multiplication classique à la différence que les indices doivent être définis comme expliqué plus haut.


### 3.4. Temps de calcul
A l'aide de l'outil <time.h>, il est possible de mesurer le temps CPU et GPU, l'addition de deux matrices de taille 500x500 fournit ainsi les résultats suivants :
- CPU : 0.909s
- GPU : 0.248s

Ces résultats confirment donc le gain de performance permis par l'utilisation du GPU.




## 4. Partie 2 - Premières couches du réseau de neurones LeNet-5 : Convolution 2D et subsampling
Cette partie 2 correspond au fichier TP_CNN.

L'architecture du réseau LeNet-5 est composée de plusieurs couches :

- Layer 1 : Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de données MNIST
- Layer 2 : Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultante est donc de 6x28x28.
- Layer 3 : Sous-échantillonnage d'un facteur 2. La taille résultante des données est donc de 6x14x14.


### 4.1. Layer 1 - Génération des données de tests
On commence par générer les données, c'est-à-dire les matrices suivantes :
- Une matrice float "raw_data" de taille 32x32 initialisé avec des valeurs comprises entre 0 et 1, correspondant aux données d'entrée
- Une matrice float "C1_data" de taille 6x28x28 initialisé à 0 qui prendra les valeurs de sortie de la convolution 2D. C1 correspond aux données après la première convolution
- Une matrice float "S1_data" de taille 6x14x14 intialisé à 0 qui prendra les valeurs de sortie du sous-échantillonnage. S1 correspond aux données après le premier sous-échantillonnage
- Une matrice float "C1_kernel" de taille 6x5x5 initialisé à des valeurs comprises entre 0 et 1 correspondant à nos premiers noyaux de convolution


### 4.2. Layer 2 - Convolution 2D
La méthode pour la convolution est similaire à la fonction de multiplication sur GPU créée précédemment, on réalise donc la convolution uniquement sur le GPU. 
Pour cela, on déplace le kernel "C1_kernel" sur la totalité de la matrice initiale "raw_data" pour obtenir la matrice résultante "C1_data".


### 4.3. Layer 3 - Sous-échantillonnage
La méthode pour le sous-échantillonage consiste en un moyennage sur une fenêtre glissante 2x2, celui-ci se fait également sur le GPU.


### 4.4. Fonction d'activation
Une couche d'activation est ensuite requise. Dans notre cas, la fonction d'activation utilisée est une tangente hyperbolique.
Celle-ci est appelée à la suite de chaque couche de convolution et retourne une matrice de même dimension que celle qui lui est fournie.




## 5. Partie 3 - Un peu de Python

### 4.1. Notebook Python
Dans cette dernière partie, le notebook Python sert référence afin de finaliser les différentes couches ainsi que les poids et biais du réseau.


### 4.2. Création des fonctions manquantes
Afin de terminer la construction du réseau, on y ajoute les couches de convolution et de sous-échantillonnage nécessaires, comme observées sur le notebook Python.
Il serait également nécessaire de créer une couche de "Dense". Mais n'ayant  pas pu importer les poids du notebook, je n'ai pas réalisé cette fonction. 


### 4.3. Export des poids dans un fichier .h5
Après avoir procédé à l'entrainement du réseau dans le notebook, l'idée est de récupérer les poids et biais de chaque couche afin d'initialiser les kernels. 


### 4.4. Bilan
N'ayant pas pu réaliser la partie précédente, je me suis contenté de tester le réseau avec des poids initialisés aléatoirement et celui-ci semble fonctionnel.





Auteur : **Louis Tonial**
