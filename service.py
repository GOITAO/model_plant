from model_prediction import predict
from Recommender import recommend

def diagnose(image_path: str):
    label, confidence = predict(image_path)
    result = recommend(label, confidence)
    result["predicted_class"] = label
    print(result)
    return result
