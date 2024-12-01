from flask import Flask, request, jsonify
from transformers import pipeline

# Utwórz aplikację Flask
app = Flask(__name__)

# Załaduj model DistilBERT
model = pipeline("text-classification", model="distilbert-base-uncased")

# Endpoint do analizy tekstu
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json  # Odbierz dane z żądania POST
    text = data.get("text")  # Pobierz tekst do analizy
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = model(text)  # Wykonaj analizę za pomocą DistilBERT
    return jsonify(result)  # Zwróć wynik analizy

# Uruchom serwer
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
