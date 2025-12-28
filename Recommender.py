import json

with open("diseases.json", encoding="utf-8") as f:
    DISEASES = json.load(f)["diseases"]

def normalize_label(label: str):
    return label.strip().rstrip("_")

def recommend(label: str, confidence: float):
    label = normalize_label(label)

    if label not in DISEASES:
        return {"status": "error", "message": "Classe inconnue"}

    disease = DISEASES[label]

    if confidence < disease["confidence_threshold"]:
        return {
            "status": "uncertain",
            "message": "Confiance insuffisante",
            "confidence": confidence
        }

    return {
        "status": "ok",
        "plant": disease["plant"],
        "disease_name": disease["disease_name"],
        "severity": disease["severity"],
        "confidence": confidence,
        "immediate_actions": disease["immediate_actions"],
        "prevention": disease["prevention"],
        "products": disease["products"],
        "weather_alert": disease["weather_alert"]
    }

if __name__ == "__main__":
    test_label = "Tomato___Early_blight"
    test_confidence = 0.92

    result = recommend(test_label, test_confidence)
    print("Résultat recommandation :")
    print(result)
