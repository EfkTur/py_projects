
************************************************************************************************

Bienvenue sur le 7e projet Openclassrooms du parcours Machine Learning

Ce fichier .txt est un guide d'utilisation des différents fichiers présents dans ce zip. La plupart des fichiers ne présentent aucune difficulté d'utilisations, nous allons donc juste préciser les modalités d'utilisation des scripts de prédictions et des utilitaires

* Predictions * 

Pour utiliser notre algorithme, l'utilisateur n'a pas besoin d'avoir darknet. Un environnement python avec OpenCV sera suffisant. 

L'utilisateur souhaitant faire une reconnaissance de personnes masqués ou non, sur des images devra:
- ajouter les poids obtenues par l'algorithme darknet dans le folder "yolo-coco-data"
- ajouter les fichiers de configuration nécessaires dans le folder "yolo-coco-data"
- ajouter l'image qu'il souhaite analyser dans le folder "images"
- mettre les bons folders sources dans le fichier yolo-3-image.py
- lancer le fichier yolo-3-image.py dans un terminal 

Pour les vidéos, les commandes sont similaires mais il faudra juste utiliser le folder videos et le fichier yolo-3-videos

A noter que les fichiers fonctionnent aussi bien avec les poids obtenues de YOLOv4 et YOLOv3

* Utils * 

Ce sont des utilitaires essentiellement pour préparer l'utilisation de dataset au format YOLO, et pouvoir lancer l'entrainement dans Google Colab. 

Le repo Git inclut des poids et des images / videos pour utilisations, y sont disponible si vous ne souhaitez pas entrainer vos modèles.

https://github.com/EfkTur/py_projects/tree/main/07_research_yolo

Lien Google Colab pour le notebook de modèles: 

https://colab.research.google.com/drive/1KjY-HrHOBfthdUF52EoFVQYgxab2eJ8X?usp=sharing



Pour toute requêtes et questions vous pouvez me contacter à l'adresse mail suivante:

turedi.efkan@gmail.com

Nous vous souhaitons une excellente journée !

Efkan

************************************************************************************************

