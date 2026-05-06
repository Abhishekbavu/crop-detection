"""
Configuration settings for the Crop Disease Prediction System (Railway Ready)
"""
import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Base configuration"""

    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # ✅ FIX: Use Railway DATABASE_URL if available, else fallback to SQLite
    DATABASE_URL = os.environ.get("DATABASE_URL")

    if DATABASE_URL:
        # Railway gives postgres:// but SQLAlchemy needs postgresql://
        SQLALCHEMY_DATABASE_URI = DATABASE_URL.replace("postgres://", "postgresql://")
    else:
        # Local fallback (works on your PC)
        SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ================= Upload Settings =================
    UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # ================= Model Settings =================
    MODEL_PATH = os.path.join(basedir, 'models', 'my_cnn_model.h5')
    IMAGE_SIZE = (150, 150)

    # ================= Crop Data =================
    CROPS = [
        'Rice', 'Wheat', 'Cotton', 'Maize', 'Sugarcane',
        'Potato', 'Tomato', 'Onion', 'Cabbage', 'Beans',
        'Barley', 'Soybean', 'Groundnut', 'Chickpea', 'Lentil',
        'Sunflower', 'Mustard', 'Sesame', 'Carrot', 'Radish',
        'Cucumber', 'Pumpkin', 'Watermelon', 'Mango', 'Banana',
        'Papaya', 'Guava', 'Orange', 'Apple', 'Grapes'
    ]

    # ================= Disease Classes =================
    DISEASE_CLASSES = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
        'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew',
        'Cherry___healthy', 'Corn___Cercospora_leaf_spot',
        'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight', 'Grape___healthy', 'Orange___Haunglongbing',
        'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
        'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
        'Strawberry___healthy', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]

    # ================= Fertilizers =================
    FERTILIZERS = {
        'nitrogen': ['Urea', 'Ammonium Nitrate', 'Ammonium Sulfate'],
        'phosphorus': ['DAP', 'SSP', 'TSP'],
        'potassium': ['MOP', 'SOP', 'Potassium Nitrate']
    }

    # ================= Disease Management =================
    DISEASE_MANAGEMENT = {
        'Tomato___Late_blight': {
            'prevention': 'Use resistant varieties, avoid overhead irrigation, good spacing',
            'treatment': 'Metalaxyl, Mancozeb, Chlorothalonil',
            'risk': 'High'
        },
        'Tomato___Early_blight': {
            'prevention': 'Crop rotation, remove infected leaves, mulching',
            'treatment': 'Chlorothalonil, Mancozeb, Azoxystrobin',
            'risk': 'Medium'
        }
        # (you can keep rest same — shortened here)
    }
