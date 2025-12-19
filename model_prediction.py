import tensorflow as tf
import numpy as np

# Charge ton modèle

# Récupère automatiquement l'ordre des classes depuis ton dataset
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    image_size=(128, 128),
    batch_size=32
)
CLASS_NAMES = training_set.class_names
print("Ordre des classes:", CLASS_NAMES)  # Pour vérifier

def predict(test_image: str):
    try:
        # Charge le modèle à chaque appel pour s'assurer d'avoir la dernière version
       model = tf.keras.models.load_model("trained_plant_disease_model.keras")
       image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
       input_arr = tf.keras.preprocessing.image.img_to_array(image)
       input_arr = np.array([input_arr]) #convert single image to batch
       predictions = model.predict(input_arr)

        # Classe prédite
       idx = int(np.argmax(predictions[0]))
       predicted_class = CLASS_NAMES[idx]

        # Confiance
       confidence = float(predictions[0][idx])

        # ✅ TOUJOURS retourner un tuple
       return predicted_class, confidence

    except Exception as e:
        print("Erreur dans predict():", e)
        return None, None
