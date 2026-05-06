"""
Configuration settings for the Crop Disease Prediction System
"""
import os

# Get the base directory
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'database', 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload settings
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model settings
    MODEL_PATH = 'models/my_cnn_model.h5'
    IMAGE_SIZE = (150, 150)
    
    # Crop data - Expanded list
    CROPS = [
        'Rice', 'Wheat', 'Cotton', 'Maize', 'Sugarcane', 
        'Potato', 'Tomato', 'Onion', 'Cabbage', 'Beans',
        'Barley', 'Soybean', 'Groundnut', 'Chickpea', 'Lentil',
        'Sunflower', 'Mustard', 'Sesame', 'Carrot', 'Radish',
        'Cucumber', 'Pumpkin', 'Watermelon', 'Mango', 'Banana',
        'Papaya', 'Guava', 'Orange', 'Apple', 'Grapes'
    ]
    
    # Disease classes (matching the trained model)
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
    
    # Fertilizer recommendations
    FERTILIZERS = {
        'nitrogen': ['Urea', 'Ammonium Nitrate', 'Ammonium Sulfate'],
        'phosphorus': ['DAP', 'SSP', 'TSP'],
        'potassium': ['MOP', 'SOP', 'Potassium Nitrate']
    }
    
    # Disease management data
    DISEASE_MANAGEMENT = {
        'Apple___Apple_scab': {
            'prevention': 'Remove fallen leaves, prune trees for air circulation, apply fungicides early in spring',
            'treatment': 'Captan, Myclobutanil, or Mancozeb fungicides',
            'risk': 'Medium'
        },
        'Apple___Black_rot': {
            'prevention': 'Remove infected fruits and mummies, improve air circulation, prune dead wood',
            'treatment': 'Mancozeb, Copper-based fungicides, Thiophanate-methyl',
            'risk': 'High'
        },
        'Apple___Cedar_apple_rust': {
            'prevention': 'Remove nearby cedar trees, apply preventive fungicides',
            'treatment': 'Myclobutanil, Propiconazole fungicides',
            'risk': 'Medium'
        },
        'Cherry___Powdery_mildew': {
            'prevention': 'Ensure good air circulation, avoid overhead watering',
            'treatment': 'Sulfur-based fungicides, Myclobutanil',
            'risk': 'Low'
        },
        'Corn___Cercospora_leaf_spot': {
            'prevention': 'Crop rotation, remove crop debris, proper spacing',
            'treatment': 'Azoxystrobin, Propiconazole fungicides',
            'risk': 'Medium'
        },
        'Corn___Common_rust': {
            'prevention': 'Plant resistant hybrids, timely planting',
            'treatment': 'Azoxystrobin, Propiconazole fungicides',
            'risk': 'Low'
        },
        'Corn___Northern_Leaf_Blight': {
            'prevention': 'Use resistant varieties, crop rotation, bury crop debris',
            'treatment': 'Pyraclostrobin, Propiconazole fungicides',
            'risk': 'High'
        },
        'Grape___Black_rot': {
            'prevention': 'Remove mummified fruits, prune for air circulation',
            'treatment': 'Mancozeb, Captan, Myclobutanil fungicides',
            'risk': 'High'
        },
        'Grape___Esca_(Black_Measles)': {
            'prevention': 'Prune during dry weather, protect pruning wounds',
            'treatment': 'No effective chemical treatment, remove infected vines',
            'risk': 'High'
        },
        'Grape___Leaf_blight': {
            'prevention': 'Remove infected leaves, improve air circulation',
            'treatment': 'Copper-based fungicides, Mancozeb',
            'risk': 'Medium'
        },
        'Orange___Haunglongbing': {
            'prevention': 'Control psyllid insects, use disease-free nursery stock',
            'treatment': 'No cure - remove infected trees, vector control',
            'risk': 'High'
        },
        'Peach___Bacterial_spot': {
            'prevention': 'Use resistant varieties, avoid overhead irrigation',
            'treatment': 'Copper-based bactericides, Oxytetracycline',
            'risk': 'Medium'
        },
        'Pepper,_bell___Bacterial_spot': {
            'prevention': 'Use pathogen-free seeds, crop rotation, drip irrigation',
            'treatment': 'Copper-based bactericides, Acibenzolar-S-methyl',
            'risk': 'Medium'
        },
        'Potato___Early_blight': {
            'prevention': 'Crop rotation, remove infected leaves, mulching',
            'treatment': 'Chlorothalonil, Mancozeb, Azoxystrobin',
            'risk': 'Medium'
        },
        'Potato___Late_blight': {
            'prevention': 'Plant resistant varieties, destroy volunteer plants',
            'treatment': 'Metalaxyl, Chlorothalonil, Mancozeb',
            'risk': 'High'
        },
        'Squash___Powdery_mildew': {
            'prevention': 'Plant resistant varieties, ensure good air circulation',
            'treatment': 'Sulfur fungicides, Potassium bicarbonate',
            'risk': 'Low'
        },
        'Strawberry___Leaf_scorch': {
            'prevention': 'Use disease-free plants, remove infected leaves',
            'treatment': 'Captan, Thiram fungicides',
            'risk': 'Low'
        },
        'Tomato___Bacterial_spot': {
            'prevention': 'Use disease-free seeds, crop rotation, avoid overhead watering',
            'treatment': 'Copper-based bactericides, Streptomycin',
            'risk': 'Medium'
        },
        'Tomato___Early_blight': {
            'prevention': 'Crop rotation, remove infected leaves, mulching',
            'treatment': 'Chlorothalonil, Mancozeb, Azoxystrobin',
            'risk': 'Medium'
        },
        'Tomato___Late_blight': {
            'prevention': 'Use resistant varieties, avoid overhead irrigation, good spacing',
            'treatment': 'Metalaxyl, Mancozeb, Chlorothalonil',
            'risk': 'High'
        },
        'Tomato___Leaf_Mold': {
            'prevention': 'Reduce humidity, improve ventilation, use resistant varieties',
            'treatment': 'Chlorothalonil, Mancozeb fungicides',
            'risk': 'Medium'
        },
        'Tomato___Septoria_leaf_spot': {
            'prevention': 'Remove infected leaves, mulch, crop rotation',
            'treatment': 'Chlorothalonil, Mancozeb, Copper fungicides',
            'risk': 'Medium'
        },
        'Tomato___Spider_mites': {
            'prevention': 'Regular monitoring, avoid water stress, natural predators',
            'treatment': 'Insecticidal soap, Neem oil, Abamectin',
            'risk': 'Low'
        },
        'Tomato___Target_Spot': {
            'prevention': 'Remove infected plant debris, avoid overhead irrigation',
            'treatment': 'Chlorothalonil, Mancozeb fungicides',
            'risk': 'Medium'
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'prevention': 'Control whitefly vectors, use resistant varieties',
            'treatment': 'No cure - remove infected plants, control whiteflies',
            'risk': 'High'
        },
        'Tomato___Tomato_mosaic_virus': {
            'prevention': 'Use virus-free seeds, sanitize tools, control aphids',
            'treatment': 'No cure - remove infected plants immediately',
            'risk': 'High'
        }
    }
