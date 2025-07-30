FROM python:3.10
WORKDIR /app

COPY requirements.txt requirements.txt
COPY app.py app.py
COPY house_pricing_model.pkl house_pricing_model.pkl
COPY preprocessing.pkl preprocessing.pkl 
COPY dataset_cleaned.csv dataset_cleaned.csv
COPY templates/ templates/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
