# Multilingual-e5-large-instruct MALAGASY - ENGLISH

Ce projet à pour objectif d'entraîner le modèle multilingual-e5-large-instruct avec les pairs MALAGASY-ENGLISH.

## Execution du script run.sh

1. Droit d'execution : **chmod +x run.sh**
2. Execution : **./run.sh**

## Les arguments script python **me5_large_instruct.py**

![image](https://github.com/user-attachments/assets/aa8b3bc3-3464-4a1f-b1d5-d3e74a95bf5f)

**--hf_token** : Token sur huggingface pour accéder les modèle et données.

**--wandb_api_key** : Api key de wandb pour stocker les logs durant l'entraînement du modèle.

**--batch_size** : Sur kaggle batch_size=16 occupe 12Go de GPU, pour un GPU assez puissant augmenter à 32, 64, ...

**--epoch** : Tour entiere des données 10
