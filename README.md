# Multilingual-e5-large-instruct MALAGASY - ENGLISH

Ce projet à pour objectif e traduire des données **anglais** en **malagasy**.

## Execution du script run.sh

1. Droit d'execution : **chmod +x run.sh**
2. Execution : **./run.sh**

## Les arguments dans le code **me5_large_instruct.py**



**--input** : Lien des données fourni dans la plateforme Hugging Face en format pandas.

**--column1** : Première colonne correspondant au format de données.

**--column2** : Deuxième colonne correspandant au format de données.

**--weave_output** : Nom du projet dans la plateforme **wandb** où les données tradutes y sont stockées.

**--batch_size** : Valeur recommendée entre 8 à 32 y compris, exemple 8, 16, 32. Augmenter selon la puissance du GPU, pour un GPU assez puissant la valeur 16 ou 32 est idéale.

**--chunk_size** : Seuil de données traduites à publier dans la plateforme **wandb**. Exemple 50, puis 50 ligne données a été traduites, le seuil 50 est atteint donc les 50 lignes données traduites seront publiées dans la plateforme **wandb**.

**--range_begin** : L'indice pour commencer la traduction

**--range_end** : L'indice pour finir la traduction
