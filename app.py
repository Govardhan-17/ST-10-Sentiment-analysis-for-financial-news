from flask import Flask, render_template, request
import pickle
import os

try:
    from test import TextToNum  # Ensure test.py exists
except ImportError:
    print("Error: test.py not found or TextToNum class is missing.")
    TextToNum = None  # Avoid crashes if test.py is missing

app = Flask(__name__)

# Define paths for model files
vectorizer_path = "vectorizer.pickle"
model_path = "model.pickle"

# Load vectorizer and model safely
if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
    print("Error: Required model files (vectorizer.pickle or model.pickle) are missing.")
    vectorizer = None
    model = None
else:
    with open(vectorizer_path, "rb") as vc:
        vectorizer = pickle.load(vc)
    with open(model_path, "rb") as mc:
        model = pickle.load(mc)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        if not TextToNum:
            return "Error: Text processing module is missing.", 500

        if not vectorizer or not model:
            return "Error: Model files are missing. Please check server logs.", 500

        msg = request.form.get("message", "").strip()
        if not msg:
            return "Error: No input text provided.", 400

        print(f"Received message: {msg}")

        # Process text using TextToNum
        ob = TextToNum(msg)
        ob.cleaner()
        ob.token()
        ob.removeStop()
        stemmed_words = ob.stemme()
        processed_text = " ".join(stemmed_words)

        # Transform input text using vectorizer
        try:
            vcdata = vectorizer.transform([processed_text]).toarray()
        except Exception as e:
            print(f"Vectorization error: {e}")
            return "Error processing input text.", 500

        # Make prediction
        try:
            pred = model.predict(vcdata)
            sentiment_map = {1: "Positive 😊", 0: "Neutral 😐", -1: "Negative 😢"}
            sentiment = sentiment_map.get(pred[0], "Unknown")
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error predicting sentiment.", 500

        return render_template("result.html", sentiment=sentiment)
    else:
        return render_template("predict.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
