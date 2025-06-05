import pickle
import pandas as pd


# Import ML model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Usually comes form MLFlow
MODEL_VERSION = 'x.x.0'

#Get the class labels from model 
class_labels = model.classes_.tolist()


def predict_output(user_input: dict):

    input_df = pd.DataFrame([user_input])

    predicted_class = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities)

    class_probs = dict(zip(class_labels, map(lambda p: round(p,4), probabilities)))

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs
    }
