# ProjetTAL
Projet de Traitement Automatique des Langues sur les données ATIS

Une version basique de atis-cnn.py est disponible sur [le site de son auteur][https://sophierosset.github.io/eidi/mini/projet/2019/01/08/eidi-atis.html]. 

## Instruction d'utilisation

Le projet comporte deux fichiers principaux : "main_train.py" et "main_test.py". Il faut dans un premier temps exécuter main_train.py pour entrainer le modèle via les données d'entrainement et de validation. Ce fichier effectue une sauvegarde du modèle ainsi que des différents vocabulaire grâce à des fichiers au format ".pkl". 

Dans le fichier "main_test.py" on charge le modèle et les vocabulaires précédemment sauvegardés afin d'effectuer la phase de test.

Il est possible d'utiliser CUDA pour exécuter le programme avec le GPU. La commande est la suivante : "python main_train.py cuda"
