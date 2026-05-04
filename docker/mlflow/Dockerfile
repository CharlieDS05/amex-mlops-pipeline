FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.16.2 \
    psycopg2-binary \
    boto3

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres/mlflow", \
     "--default-artifact-root", "s3://mlflow/", \
     "--host", "0.0.0.0", \
     "--port", "5000"]