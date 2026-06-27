# Usine de Fine-Tuning de SLM Financier (MLOps)

Une architecture Machine Learning de bout en bout, du versioning des données jusqu'au fine-tuning d'un Small Language Model (SLM) et son déploiement web. Ce projet démontre la mise en place d'un pipeline MLOps complet pour spécialiser un modèle de langage puissant (Llama-3) sur des tâches d'analyse de sentiment financier, le tout orchestré sur l'écosystème Google Cloud et Hugging Face.

- **Lien du Modèle :** [Registre Hugging Face](https://huggingface.co/FakhAhmed/slm-finance-sentiment)

## 🚀 Fonctionnalités Clés
- **Fine-Tuning PEFT / LoRA :** Spécialisation du modèle fondation Meta Llama-3 (8B) sans ré-entraîner tous les poids. Utilisation de la quantification 4-bit (via Unsloth et BitsAndBytes) pour une optimisation drastique de la VRAM, permettant l'entraînement sur un simple GPU T4.
- **Versioning des Données (DVC) :** La donnée d'entraînement financière est traitée comme du code. Son historique est strictement versionné via Data Version Control (DVC) avec un stockage distant et sécurisé sur Google Cloud Storage.
- **Tracking d'Expérimentations (MLflow) :** Déploiement d'un serveur de tracking MLflow propriétaire sur Google Cloud Run (Serverless), adossé à une base de données Cloud SQL (PostgreSQL). Chaque métrique d'entraînement (Loss) et hyperparamètre est enregistré en direct dans le cloud.
- **Inférence Interactive (Hugging Face Spaces) :** Le modèle fine-tuné est hébergé publiquement et interrogé via une interface web fluide développée avec Gradio, prouvant la capacité d'industrialisation et de "Serving" du modèle.

## 🏗️ Architecture Technique
- **Modèle Fondation :** Meta Llama-3 (8B)
- **Entraînement Machine Learning :** PyTorch, Hugging Face Transformers, PEFT (LoRA), Unsloth, TRL
- **Versioning Data :** DVC (Data Version Control)
- **Tracking MLOps :** MLflow
- **Cloud Infrastructure (GCP) :** Google Cloud Run, Cloud SQL (PostgreSQL), Cloud Storage
- **Déploiement & Registre :** Hugging Face Hub, Hugging Face Spaces, Gradio

## 📁 Structure du Projet
- **`data/`** : Espace de stockage des jeux de données d'entraînement (données financières brutes et processées), trackés et versionnés par DVC.
- **`notebooks/`** : Laboratoire d'entraînement central (`finetuning_slm.ipynb`). Ce notebook, orchestré sur Google Colab, exécute le fine-tuning Unsloth et pousse les métriques vers le serveur MLflow distant.
- **`src/`** : Contient le code source de l'interface web (`app.py`). Il charge le modèle de base, fusionne dynamiquement l'adaptateur LoRA téléchargé depuis le cloud, et génère l'interface utilisateur Gradio.
- **`dvc.yaml`** : Le cerveau du pipeline de données, définissant l'orchestration et le tracking du dataset vers Google Cloud Storage.
- **`requirements.txt`** : Liste exhaustive des dépendances Python pour la reproductibilité de l'environnement de développement local.
- **`requirements_app.txt`** : Recette de dépendances allégée, spécifiquement conçue pour garantir un déploiement rapide et sans conflit sur le conteneur Hugging Face Spaces.

## 💡 Pourquoi ce projet ?
Ce PoC (Proof of Concept) illustre la complexité réelle de l'ingénierie autour des Large Language Models (LLMOps). Il prouve la capacité à dépasser le simple script Python pour construire une véritable "usine" logicielle : capable de gérer des données versionnées, d'optimiser l'entraînement de milliards de paramètres sur des ressources matérielles contraintes, et de traquer rigoureusement les expérimentations sur le Cloud avant un déploiement accessible au public.