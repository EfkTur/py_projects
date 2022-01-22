-- Bienvenue sour le projet Stack_Tags de Efkan Turedi --

Ce fichier readme.txt donne quelques précisions concernant les fichiers présents dans ce dossier.

	- Le fichier "Stackoverflow_tags_notebook" est le notebook principal à utiliser
	
	- Le notebook "50k_Stackoverflow_tags_notebook" est essentiellement un notebook support qui nous a été utile pour itérer rapidement sur une base de données plus petite. A noter que certains graphiques de notre présentation ne sont disponible que dans ce notebook
	
	- La partie du notebook concernant les reseaux de Neurones se trouvent dans ce lien:
	https://colab.research.google.com/drive/1oHB_lZ8EL7vFAZZqIzkDIcBDFyUSDOkZ?usp=sharing

	- Le rapport et la présentation sont au format pdf dans ce dossier 

	- Le folder modelNN n'est pas présent dans ce dossier car trop gros. Mais il est 		possible de le générer simplement dans le notebook Google Colab. Ensuite, il faudra 	fournir ce folder en input à mlflow avec la commande suivante sur votre oridnateur 		en local:  mlflow models serve -m modelNN 
	(pour faire tourner l'API)
		
	- Le dashboard est déployé avec le fichier dashboard.py et la commande:
	streamlit run dashboard.py
	
	- Le tokenizer et MiltiLabelBinarizer exporté du reseau de neurones sont présents
	
	
Vous pouvez me contacter via mon adresse mail pour toute questions ou requêtes: 		turedi.efkan@gmail.com

Bien à vous, 

Efkan
