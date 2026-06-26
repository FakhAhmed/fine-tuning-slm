FROM python:3.10-slim

# On fixe la version de MLflow à la 2.8.1 pour éviter le blocage DNS
RUN pip install mlflow==2.8.1 psycopg2-binary google-cloud-storage

EXPOSE 5000
