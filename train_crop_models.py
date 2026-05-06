"""
Train multiple ML models for crop recommendation
Implements Decision Tree, Random Forest, and Logistic Regression

Dataset: Uses Kaggle Crop Recommendation Dataset if available
Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
Save as: Crop_recommendation.csv in project root
"""
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_kaggle_dataset(filepath='Crop_recommendation.csv'):
    """
    Load the Kaggle Crop Recommendation Dataset
    
    Dataset columns: N, P, K, temperature, humidity, ph, rainfall, label
    Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with crop recommendation data or None if not found
    """
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"   ✓ Loaded Kaggle dataset from {filepath}")
            print(f"   Dataset shape: {df.shape}")
            print(f"   Crops in dataset: {df['label'].nunique()}")
            
            # Rename 'label' column to 'crop' for consistency
            if 'label' in df.columns:
                df = df.rename(columns={'label': 'crop'})
            
            # We'll use N, P, K, temperature, humidity, rainfall (ignoring pH for now)
            return df[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall', 'crop']]
        except Exception as e:
            print(f"   [ERROR] Failed to load dataset: {str(e)}")
            return None
    return None

def create_dummy_crop_dataset():
    """
    Create a dummy crop recommendation dataset with expanded crop list
    For production, use real datasets like Crop Recommendation Dataset from Kaggle
    """
    np.random.seed(42)
    
    # Define crops and their ideal conditions - EXPANDED LIST
    crops_data = {
        # Original crops
        'Rice': {'N': (80, 100), 'P': (50, 70), 'K': (40, 50), 'temp': (20, 35), 'humidity': (70, 90), 'rainfall': (150, 300)},
        'Wheat': {'N': (50, 70), 'P': (40, 60), 'K': (30, 40), 'temp': (15, 25), 'humidity': (40, 60), 'rainfall': (50, 100)},
        'Cotton': {'N': (50, 80), 'P': (40, 60), 'K': (30, 50), 'temp': (20, 30), 'humidity': (40, 70), 'rainfall': (80, 150)},
        'Maize': {'N': (60, 90), 'P': (40, 60), 'K': (30, 40), 'temp': (18, 27), 'humidity': (50, 70), 'rainfall': (100, 150)},
        'Potato': {'N': (50, 80), 'P': (50, 80), 'K': (50, 80), 'temp': (15, 25), 'humidity': (60, 80), 'rainfall': (70, 100)},
        'Tomato': {'N': (60, 85), 'P': (50, 70), 'K': (40, 60), 'temp': (20, 30), 'humidity': (65, 85), 'rainfall': (90, 130)},
        'Sugarcane': {'N': (70, 100), 'P': (30, 50), 'K': (40, 60), 'temp': (25, 35), 'humidity': (60, 85), 'rainfall': (120, 200)},
        'Cabbage': {'N': (60, 80), 'P': (40, 60), 'K': (50, 70), 'temp': (15, 25), 'humidity': (60, 80), 'rainfall': (60, 100)},
        'Onion': {'N': (40, 70), 'P': (30, 50), 'K': (40, 60), 'temp': (18, 28), 'humidity': (50, 70), 'rainfall': (50, 90)},
        'Beans': {'N': (30, 60), 'P': (40, 70), 'K': (30, 50), 'temp': (18, 28), 'humidity': (60, 80), 'rainfall': (70, 120)},
        # New crops added
        'Barley': {'N': (40, 70), 'P': (35, 55), 'K': (25, 40), 'temp': (12, 22), 'humidity': (35, 55), 'rainfall': (40, 80)},
        'Soybean': {'N': (20, 50), 'P': (40, 70), 'K': (35, 55), 'temp': (20, 30), 'humidity': (55, 75), 'rainfall': (80, 140)},
        'Groundnut': {'N': (20, 50), 'P': (35, 60), 'K': (40, 65), 'temp': (22, 32), 'humidity': (50, 70), 'rainfall': (60, 110)},
        'Chickpea': {'N': (20, 45), 'P': (35, 60), 'K': (30, 50), 'temp': (20, 28), 'humidity': (40, 60), 'rainfall': (40, 70)},
        'Lentil': {'N': (20, 45), 'P': (40, 65), 'K': (25, 45), 'temp': (18, 27), 'humidity': (45, 65), 'rainfall': (35, 65)},
        'Sunflower': {'N': (60, 90), 'P': (40, 65), 'K': (50, 75), 'temp': (20, 30), 'humidity': (40, 70), 'rainfall': (60, 100)},
        'Mustard': {'N': (50, 80), 'P': (30, 55), 'K': (35, 55), 'temp': (18, 25), 'humidity': (45, 65), 'rainfall': (40, 75)},
        'Sesame': {'N': (40, 70), 'P': (30, 55), 'K': (35, 60), 'temp': (25, 35), 'humidity': (45, 65), 'rainfall': (50, 90)},
        'Carrot': {'N': (50, 75), 'P': (45, 70), 'K': (50, 75), 'temp': (15, 25), 'humidity': (55, 75), 'rainfall': (60, 95)},
        'Radish': {'N': (45, 70), 'P': (40, 65), 'K': (45, 70), 'temp': (15, 25), 'humidity': (55, 75), 'rainfall': (50, 85)},
        'Cucumber': {'N': (50, 80), 'P': (45, 70), 'K': (45, 70), 'temp': (20, 30), 'humidity': (60, 80), 'rainfall': (70, 110)},
        'Pumpkin': {'N': (55, 85), 'P': (50, 75), 'K': (50, 75), 'temp': (22, 32), 'humidity': (55, 75), 'rainfall': (80, 120)},
        'Watermelon': {'N': (50, 80), 'P': (45, 70), 'K': (50, 75), 'temp': (24, 34), 'humidity': (50, 70), 'rainfall': (70, 115)},
        'Mango': {'N': (60, 90), 'P': (35, 60), 'K': (50, 80), 'temp': (24, 35), 'humidity': (50, 75), 'rainfall': (100, 180)},
        'Banana': {'N': (65, 95), 'P': (40, 65), 'K': (60, 90), 'temp': (22, 32), 'humidity': (65, 85), 'rainfall': (120, 220)},
        'Papaya': {'N': (55, 85), 'P': (35, 60), 'K': (45, 70), 'temp': (24, 34), 'humidity': (60, 80), 'rainfall': (90, 150)},
        'Guava': {'N': (50, 80), 'P': (35, 60), 'K': (50, 75), 'temp': (23, 33), 'humidity': (55, 75), 'rainfall': (80, 140)},
        'Orange': {'N': (55, 85), 'P': (35, 60), 'K': (45, 70), 'temp': (20, 32), 'humidity': (50, 75), 'rainfall': (90, 160)},
        'Apple': {'N': (50, 75), 'P': (30, 55), 'K': (40, 65), 'temp': (10, 25), 'humidity': (50, 70), 'rainfall': (80, 140)},
        'Grapes': {'N': (45, 75), 'P': (35, 60), 'K': (50, 75), 'temp': (15, 30), 'humidity': (45, 70), 'rainfall': (70, 120)}
    }
    
    # Generate 150 samples per crop for better accuracy (4500 total samples)
    data = []
    for crop, params in crops_data.items():
        for _ in range(150):
            sample = {
                'N': np.random.uniform(params['N'][0], params['N'][1]),
                'P': np.random.uniform(params['P'][0], params['P'][1]),
                'K': np.random.uniform(params['K'][0], params['K'][1]),
                'temperature': np.random.uniform(params['temp'][0], params['temp'][1]),
                'humidity': np.random.uniform(params['humidity'][0], params['humidity'][1]),
                'rainfall': np.random.uniform(params['rainfall'][0], params['rainfall'][1]),
                'crop': crop
            }
            data.append(sample)
    
    df = pd.DataFrame(data)
    return df

def train_models():
    """Train multiple ML models and save them"""
    print("="*60)
    print("Crop Recommendation ML Model Training")
    print("="*60)
    
    # Try to load Kaggle dataset first
    print("\n1. Loading dataset...")
    df = load_kaggle_dataset()
    
    if df is None:
        print("   ⚠ Kaggle dataset not found!")
        print("   ")
        print("   To use real data, download from:")
        print("   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        print("   Save as: Crop_recommendation.csv in project root")
        print("   ")
        print("   Using dummy dataset for training...")
        df = create_dummy_crop_dataset()
    print(f"   Dataset created with {len(df)} samples")
    print(f"   Crops: {df['crop'].unique().tolist()}")
    
    # Prepare data
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]
    y = df['crop']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n2. Data split: {len(X_train)} training, {len(X_test)} testing samples")
    
    # Feature Scaling for better accuracy
    print("\n3. Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   Features normalized for improved model performance")
    
    # Create models directory
    os.makedirs('models/crop_models', exist_ok=True)
    
    # Save scaler for later use
    scaler_file = 'models/crop_models/scaler.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved to: {scaler_file}")
    
    # Optimized models with better hyperparameters
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=20, min_samples_split=4, min_samples_leaf=2, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, multi_class='ovr', solver='lbfgs', C=10)
    }
    
    results = {}
    
    print("\n4. Training models with optimized parameters...")
    print("-"*60)
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Train model with scaled data
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Save model
        model_filename = f"models/crop_models/{name.lower().replace(' ', '_')}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        # Store results
        results[name] = {
            'accuracy': accuracy * 100,
            'cv_mean': cv_scores.mean() * 100,
            'cv_std': cv_scores.std() * 100,
            'model_file': model_filename
        }
        
        print(f"   ✓ {name} trained successfully")
        print(f"     - Accuracy: {accuracy*100:.2f}%")
        print(f"     - CV Score: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
        print(f"     - Saved to: {model_filename}")
    
    # Save results summary
    results_file = 'models/crop_models/model_metrics.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*60)
    print("Model Training Complete!")
    print("="*60)
    print("\nModel Performance Comparison:")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:20s} | Accuracy: {metrics['accuracy']:.2f}% | CV: {metrics['cv_mean']:.2f}%")
    print("-"*60)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]} ({best_model[1]['accuracy']:.2f}%)")
    
    return results

if __name__ == '__main__':
    try:
        results = train_models()
        print("\n✓ All models saved successfully!")
        print("  Use these models in the Flask application for predictions.")
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
