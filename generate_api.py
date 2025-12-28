from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from typing import Dict
import uvicorn
from Recommender import recommend

app = FastAPI(
    title="Plant Disease Detection API",
    description="API pour détecter les maladies des plantes à partir d'images",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
MODEL = None
CLASS_NAMES = None

# Charger le modèle et les classes au démarrage
@app.on_event("startup")
async def load_model():
    global MODEL, CLASS_NAMES
    try:
        # Charge le modèle
        MODEL = tf.keras.models.load_model("trained_plant_disease_model.keras")
        print("✅ Modèle chargé avec succès")
        
        # Récupère les noms de classes
        training_set = tf.keras.utils.image_dataset_from_directory(
            'train',
            labels="inferred",
            label_mode="categorical",
            image_size=(128, 128),
            batch_size=32
        )
        CLASS_NAMES = training_set.class_names
        print(f"✅ Classes détectées: {CLASS_NAMES}")
        
    except Exception as e:
        print(f"❌ Erreur au chargement: {e}")
        raise


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Prétraite l'image pour la prédiction"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertit en RGB si nécessaire
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Redimensionne à 128x128
        image = image.resize((128, 128))
        
        # Convertit en array numpy
        input_arr = np.array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        
        return input_arr
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de traitement d'image: {str(e)}")


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "API de détection de maladies de plantes",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Prédire la maladie d'une plante",
            "/health": "GET - Vérifier l'état de l'API",
            "/classes": "GET - Obtenir la liste des classes",
            "/docs": "GET - Documentation interactive"
        }
    }


@app.get("/health")
async def health_check():
    """Vérifie l'état de l'API et du modèle"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "classes_loaded": CLASS_NAMES is not None,
        "num_classes": len(CLASS_NAMES) if CLASS_NAMES else 0
    }


@app.get("/classes")
async def get_classes():
    """Retourne la liste des classes disponibles"""
    if CLASS_NAMES is None:
        raise HTTPException(status_code=503, detail="Classes non chargées")
    
    return {
        "classes": CLASS_NAMES,
        "total": len(CLASS_NAMES)
    }



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None or CLASS_NAMES is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        # Lire et prétraiter l'image
        image_bytes = await file.read()
        input_arr = preprocess_image(image_bytes)
        
        # Faire la prédiction
        predictions = MODEL.predict(input_arr, verbose=0)
        idx = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[idx]
        confidence = float(predictions[0][idx])
        
        # Obtenir les recommandations basées sur le label prédit
        rec = recommend(predicted_class, confidence)
        
        # Vérifier le status de la recommandation
        if rec["status"] != "ok":
            return {
                "success": False,
                "message": rec.get("message", "Erreur inconnue"),
                "confidence": confidence
            }
        
        # Construire la réponse finale pour le frontend
        response = {
            "success": True,
            "plant": rec["plant"],
            "disease": rec["disease_name"],
            "severity": rec["severity"],
            "confidence": round(confidence * 100, 2),
            "description": rec.get("description", ""),
            "recommendations": rec.get("immediate_actions", []),
            "prevention": rec.get("prevention", ""),
            "products": rec.get("products", []),
            "weather_alert": rec.get("weather_alert", ""),
            "filename": file.filename
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Prédit les maladies pour plusieurs images
    
    Args:
        files: Liste d'images
    
    Returns:
        JSON avec les prédictions pour chaque image
    """
    if MODEL is None or CLASS_NAMES is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images par requête")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Format de fichier invalide"
            })
            continue
        
        try:
            image_bytes = await file.read()
            input_arr = preprocess_image(image_bytes)
            predictions = MODEL.predict(input_arr, verbose=0)
            
            idx = int(np.argmax(predictions[0]))
            predicted_class = CLASS_NAMES[idx]
            confidence = float(predictions[0][idx])
            
            results.append({
                "filename": file.filename,
                "success": True,
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total_images": len(files),
        "results": results
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)