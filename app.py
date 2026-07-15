"""
AI-Driven Crop Disease Prediction and Management System
Main Flask Application
"""
import os
import pickle
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename
import bcrypt
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy

from config import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db = SQLAlchemy(app)

# Global variables for ML models
disease_model = None
crop_models = {}
model_metrics = {}
scaler = None


# ==================== Database Models ====================

class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='farmer')  # farmer or admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    soil_data = db.relationship('SoilData', backref='user', lazy=True)
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        """Verify password"""
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


class SoilData(db.Model):
    """Soil and weather data model"""
    __tablename__ = 'soil_data'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PredictionHistory(db.Model):
    """Prediction history model"""
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)  # crop/fertilizer/disease
    result = db.Column(db.String(500), nullable=False)
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ==================== Helper Functions ====================

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if user.role != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_disease_model():
    """Load the TensorFlow disease detection model"""
    global disease_model
    try:
        import tensorflow as tf
        model_path = app.config['MODEL_PATH']
        if os.path.exists(model_path):
            disease_model = tf.keras.models.load_model(model_path)
            print(f"✓ Disease detection model loaded successfully from {model_path}")
        else:
            print(f"⚠ Warning: Model file not found at {model_path}")
            print("  Disease detection will use simulation mode")
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        print("  Disease detection will use simulation mode")


def load_crop_models():
    """Load crop recommendation ML models"""
    global crop_models, model_metrics, scaler
    models_dir = 'models/crop_models'
    
    if not os.path.exists(models_dir):
        print(f"[WARNING] Crop models directory not found at {models_dir}")
        print("  Run 'python train_crop_models.py' to train models first")
        print("  Crop recommendation will use rule-based approach")
        return
    
    try:
        # Load trained models
        model_files = {
            'Decision Tree': 'decision_tree_model.pkl',
            'Random Forest': 'random_forest_model.pkl',
            'Logistic Regression': 'logistic_regression_model.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    crop_models[name] = pickle.load(f)
                print(f"[OK] {name} model loaded")
        
        # Load scaler
        scaler_file = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            print(f"[OK] Feature scaler loaded")
        
        # Load metrics
        metrics_file = os.path.join(models_dir, 'model_metrics.pkl')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'rb') as f:
                model_metrics = pickle.load(f)
            print(f"[OK] Model metrics loaded")
        
        if crop_models:
            print(f"[OK] {len(crop_models)} crop recommendation models loaded successfully")
        
    except Exception as e:
        print(f"[ERROR] Error loading crop models: {str(e)}")
        print("  Crop recommendation will use rule-based approach")


def recommend_crop(n, p, k, temp, humidity, rainfall, season=None):
    """Recommend crops using ML models or rule-based approach"""
    global crop_models, scaler
    
    # Prepare input data
    input_data = np.array([[n, p, k, temp, humidity, rainfall]])
    
    # Try ML-based prediction first
    if crop_models:
        try:
            # Use Random Forest (usually best performing)
            model = crop_models.get('Random Forest') or list(crop_models.values())[0]
            
            # Scale input data if scaler available
            if scaler is not None:
                input_data_scaled = scaler.transform(input_data)
            else:
                input_data_scaled = input_data
            
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(input_data_scaled)[0]
                classes = model.classes_
                
                # Get top 3 predictions
                top_indices = np.argsort(probas)[::-1][:3]
                recommendations = [(classes[i], probas[i] * 100) for i in top_indices]
                
                # Filter by season if provided
                if season:
                    season_crops = get_seasonal_crops(season)
                    recommendations = [(crop, score) for crop, score in recommendations 
                                     if crop in season_crops]
                
                return recommendations if recommendations else get_rule_based_recommendations(n, p, k, temp, humidity, rainfall)
            
        except Exception as e:
            print(f"ML prediction error: {str(e)}")
    
    # Fallback to rule-based recommendation
    return get_rule_based_recommendations(n, p, k, temp, humidity, rainfall)


def get_seasonal_crops(season):
    """Return crops suitable for each season - EXPANDED"""
    seasonal_mapping = {
        'Kharif': ['Rice', 'Maize', 'Cotton', 'Sugarcane', 'Tomato', 'Soybean', 'Groundnut', 
                   'Sunflower', 'Sesame', 'Cucumber', 'Pumpkin', 'Watermelon'],  # June-October (Monsoon)
        'Rabi': ['Wheat', 'Potato', 'Onion', 'Cabbage', 'Barley', 'Chickpea', 'Lentil', 
                 'Mustard', 'Carrot', 'Radish'],  # October-March (Winter)
        'Zaid': ['Tomato', 'Beans', 'Onion', 'Cucumber', 'Watermelon', 'Pumpkin', 
                 'Mango', 'Papaya', 'Guava'],  # March-June (Summer)
        'All Season': app.config['CROPS']
    }
    return seasonal_mapping.get(season, app.config['CROPS'])


def get_rule_based_recommendations(n, p, k, temp, humidity, rainfall):
    """Rule-based crop recommendation as fallback"""
    crops_scores = {}
    
    # Rice
    if 80 <= n <= 100 and 50 <= p <= 70 and 40 <= k <= 50 and 20 <= temp <= 35 and rainfall > 150:
        crops_scores['Rice'] = 85 + np.random.randint(0, 10)
    
    # Wheat
    if 50 <= n <= 70 and 40 <= p <= 60 and 30 <= k <= 40 and 15 <= temp <= 25 and 50 <= rainfall <= 100:
        crops_scores['Wheat'] = 80 + np.random.randint(0, 15)
    
    # Cotton
    if 50 <= n <= 80 and 40 <= p <= 60 and 30 <= k <= 50 and 20 <= temp <= 30:
        crops_scores['Cotton'] = 75 + np.random.randint(0, 15)
    
    # Maize
    if 60 <= n <= 90 and 40 <= p <= 60 and 30 <= k <= 40 and 18 <= temp <= 27:
        crops_scores['Maize'] = 80 + np.random.randint(0, 12)
    
    # Potato
    if 50 <= n <= 80 and 50 <= p <= 80 and 50 <= k <= 80 and 15 <= temp <= 25:
        crops_scores['Potato'] = 82 + np.random.randint(0, 10)
    
    # Tomato
    if 60 <= n <= 85 and 50 <= p <= 70 and 40 <= k <= 60 and 20 <= temp <= 30:
        crops_scores['Tomato'] = 78 + np.random.randint(0, 12)
    
    # Default recommendations if no specific match
    if not crops_scores:
        all_crops = app.config['CROPS']
        for crop in np.random.choice(all_crops, 3, replace=False):
            crops_scores[crop] = 60 + np.random.randint(0, 20)
    
    # Sort and return top 3
    sorted_crops = sorted(crops_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return sorted_crops


def recommend_fertilizer(n, p, k, crop):
    """Recommend fertilizer based on NPK values and crop"""
    # Optimal NPK ranges for different crops
    optimal_npk = {
        'Rice': {'N': 80, 'P': 60, 'K': 40},
        'Wheat': {'N': 60, 'P': 50, 'K': 35},
        'Cotton': {'N': 70, 'P': 50, 'K': 40},
        'Maize': {'N': 75, 'P': 50, 'K': 35},
        'Potato': {'N': 70, 'P': 70, 'K': 70},
        'Tomato': {'N': 75, 'P': 60, 'K': 50},
    }
    
    optimal = optimal_npk.get(crop, {'N': 70, 'P': 55, 'K': 40})
    
    recommendations = []
    
    # Check Nitrogen
    n_diff = optimal['N'] - n
    if n_diff > 20:
        recommendations.append({
            'nutrient': 'Nitrogen',
            'status': 'Low',
            'fertilizer': 'Urea',
            'dosage': f'{abs(n_diff) * 2} kg/acre'
        })
    elif n_diff < -20:
        recommendations.append({
            'nutrient': 'Nitrogen',
            'status': 'High',
            'fertilizer': 'None',
            'dosage': 'Reduce nitrogen-based fertilizers'
        })
    
    # Check Phosphorus
    p_diff = optimal['P'] - p
    if p_diff > 15:
        recommendations.append({
            'nutrient': 'Phosphorus',
            'status': 'Low',
            'fertilizer': 'DAP (Di-Ammonium Phosphate)',
            'dosage': f'{abs(p_diff) * 2.5} kg/acre'
        })
    elif p_diff < -15:
        recommendations.append({
            'nutrient': 'Phosphorus',
            'status': 'High',
            'fertilizer': 'None',
            'dosage': 'Reduce phosphorus-based fertilizers'
        })
    
    # Check Potassium
    k_diff = optimal['K'] - k
    if k_diff > 15:
        recommendations.append({
            'nutrient': 'Potassium',
            'status': 'Low',
            'fertilizer': 'MOP (Muriate of Potash)',
            'dosage': f'{abs(k_diff) * 2} kg/acre'
        })
    elif k_diff < -15:
        recommendations.append({
            'nutrient': 'Potassium',
            'status': 'High',
            'fertilizer': 'None',
            'dosage': 'Reduce potassium-based fertilizers'
        })
    
    if not recommendations:
        recommendations.append({
            'nutrient': 'All nutrients',
            'status': 'Optimal',
            'fertilizer': 'Balanced NPK (10:10:10)',
            'dosage': '50 kg/acre for maintenance'
        })
    
    return recommendations


def predict_disease(image_path):
    """Predict plant disease from image with confidence threshold"""
    global disease_model
    
    try:
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.resize(app.config['IMAGE_SIZE'])
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict using model if available
        if disease_model is not None:
            predictions = disease_model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx]) * 100
            
            disease_classes = app.config['DISEASE_CLASSES']
            
            # Check if confidence is too low (image might not be from dataset)
            if confidence < 50.0:
                return {
                    'disease': 'Uncertain Detection',
                    'raw_class': 'Unknown',
                    'confidence': round(confidence, 1),
                    'status': 'Low Confidence - Image may not be a plant leaf or is unclear. Please upload a clearer image of the plant leaf.'
                }
            
            predicted_class = disease_classes[predicted_class_idx] if predicted_class_idx < len(disease_classes) else 'Unknown'
        else:
            # Simulation mode
            disease_classes = app.config['DISEASE_CLASSES']
            predicted_class = np.random.choice(disease_classes)
            confidence = 75.0 + (np.random.random() * 20.0)
        
        # Format disease name for display (remove ___ and make readable)
        display_name = predicted_class.replace('___', ' - ').replace('_', ' ')
        
        # Determine if healthy or diseased
        is_healthy = 'healthy' in predicted_class.lower()
        
        return {
            'disease': display_name,
            'raw_class': predicted_class,  # Keep original for management lookup
            'confidence': round(confidence, 1),  # Changed to 1 decimal place for better display
            'status': 'Healthy' if is_healthy else 'Diseased'
        }
    
    except Exception as e:
        print(f"Error in disease prediction: {str(e)}")
        return {
            'disease': 'Error',
            'confidence': 0,
            'status': 'Error processing image'
        }


# ==================== Routes ====================

@app.route('/')
def index():
    """Home page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role', 'farmer')
        
        # Validate inputs
        if not all([name, email, password]):
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))
        
        # Check if user exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(name=name, email=email, role=role)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_role'] = user.role
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard - redirects based on role"""
    user = User.query.get(session['user_id'])
    
    # Redirect admin to admin dashboard
    if user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    
    # Farmer dashboard
    # Get recent predictions
    recent_predictions = PredictionHistory.query.filter_by(user_id=user.id)\
        .order_by(PredictionHistory.created_at.desc()).limit(5).all()
    
    # Get statistics
    total_predictions = PredictionHistory.query.filter_by(user_id=user.id).count()
    crop_predictions = PredictionHistory.query.filter_by(user_id=user.id, prediction_type='crop').count()
    disease_predictions = PredictionHistory.query.filter_by(user_id=user.id, prediction_type='disease').count()
    fertilizer_predictions = PredictionHistory.query.filter_by(user_id=user.id, prediction_type='fertilizer').count()
    
    stats = {
        'total': total_predictions,
        'crop': crop_predictions,
        'disease': disease_predictions,
        'fertilizer': fertilizer_predictions
    }
    
    return render_template('dashboard.html', user=user, predictions=recent_predictions, stats=stats, model_metrics=model_metrics)


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard with system-wide statistics"""
    # Get all users
    total_users = User.query.count()
    farmers = User.query.filter_by(role='farmer').count()
    admins = User.query.filter_by(role='admin').count()
    
    # Get system-wide predictions
    total_predictions = PredictionHistory.query.count()
    crop_predictions = PredictionHistory.query.filter_by(prediction_type='crop').count()
    disease_predictions = PredictionHistory.query.filter_by(prediction_type='disease').count()
    fertilizer_predictions = PredictionHistory.query.filter_by(prediction_type='fertilizer').count()
    
    # Get recent system activity
    recent_activity = PredictionHistory.query\
        .order_by(PredictionHistory.created_at.desc()).limit(10).all()
    
    # Get all users for management
    all_users = User.query.order_by(User.created_at.desc()).all()
    
    stats = {
        'total_users': total_users,
        'farmers': farmers,
        'admins': admins,
        'predictions': {
            'total': total_predictions,
            'crop': crop_predictions,
            'disease': disease_predictions,
            'fertilizer': fertilizer_predictions
        }
    }
    
    return render_template('admin_dashboard.html', 
                         stats=stats, 
                         recent_activity=recent_activity,
                         users=all_users,
                         model_metrics=model_metrics)


@app.route('/crop', methods=['GET', 'POST'])
@login_required
def crop_recommendation():
    """Crop recommendation page"""
    if request.method == 'POST':
        try:
            # Get form data
            n = float(request.form.get('nitrogen'))
            p = float(request.form.get('phosphorus'))
            k = float(request.form.get('potassium'))
            temp = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            rainfall = float(request.form.get('rainfall'))
            season = request.form.get('season', 'All Season')
            
            # Validate ranges
            if not (0 <= n <= 200 and 0 <= p <= 200 and 0 <= k <= 200):
                flash('NPK values must be between 0 and 200.', 'danger')
                return redirect(url_for('crop_recommendation'))
            
            # Save soil data
            soil_data = SoilData(
                user_id=session['user_id'],
                nitrogen=n,
                phosphorus=p,
                potassium=k,
                temperature=temp,
                humidity=humidity,
                rainfall=rainfall
            )
            db.session.add(soil_data)
            
            # Get recommendations (with season)
            recommendations = recommend_crop(n, p, k, temp, humidity, rainfall, season)
            
            # Determine which model was used
            model_used = None
            model_metrics_used = None
            if crop_models:
                # Use best performing model
                best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
                model_used = best_model[0]
                model_metrics_used = best_model[1]
            
            # Save prediction
            result_text = ', '.join([f"{crop} ({score:.1f}%)" for crop, score in recommendations])
            prediction = PredictionHistory(
                user_id=session['user_id'],
                prediction_type='crop',
                result=result_text,
                confidence=recommendations[0][1] if recommendations else 0
            )
            db.session.add(prediction)
            db.session.commit()
            
            return render_template('result.html', 
                                   result_type='crop',
                                   recommendations=recommendations,
                                   model_used=model_used,
                                   model_metrics=model_metrics_used)
        
        except ValueError:
            flash('Please enter valid numeric values.', 'danger')
            return redirect(url_for('crop_recommendation'))
    
    return render_template('crop.html')


@app.route('/fertilizer', methods=['GET', 'POST'])
@login_required
def fertilizer_recommendation():
    """Fertilizer recommendation page"""
    if request.method == 'POST':
        try:
            # Get form data
            n = float(request.form.get('nitrogen'))
            p = float(request.form.get('phosphorus'))
            k = float(request.form.get('potassium'))
            crop = request.form.get('crop')
            
            # Get recommendations
            recommendations = recommend_fertilizer(n, p, k, crop)
            
            # Save prediction
            result_text = f"Fertilizer for {crop}: " + ', '.join([f"{r['fertilizer']}" for r in recommendations])
            prediction = PredictionHistory(
                user_id=session['user_id'],
                prediction_type='fertilizer',
                result=result_text
            )
            db.session.add(prediction)
            db.session.commit()
            
            return render_template('result.html',
                                   result_type='fertilizer',
                                   crop=crop,
                                   fertilizer_recommendations=recommendations)
        
        except ValueError:
            flash('Please enter valid numeric values.', 'danger')
            return redirect(url_for('fertilizer_recommendation'))
    
    crops = app.config['CROPS']
    return render_template('fertilizer.html', crops=crops)


@app.route('/disease', methods=['GET', 'POST'])
@login_required
def disease_detection():
    """Disease detection page"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'plant_image' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(url_for('disease_detection'))
        
        file = request.files['plant_image']
        
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('disease_detection'))
        
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict disease
            prediction = predict_disease(filepath)
            
            # Get disease management info (use raw class name for lookup)
            raw_disease_class = prediction.get('raw_class', prediction['disease'])
            management_info = app.config['DISEASE_MANAGEMENT'].get(raw_disease_class, {
                'prevention': 'Maintain good agricultural practices, monitor plants regularly',
                'treatment': 'Consult local agricultural expert for specific treatment',
                'risk': 'Unknown'
            })
            
            # Save prediction
            prediction_record = PredictionHistory(
                user_id=session['user_id'],
                prediction_type='disease',
                result=f"{prediction['disease']} ({prediction['confidence']}%)",
                confidence=prediction['confidence']
            )
            db.session.add(prediction_record)
            db.session.commit()
            
            return render_template('result.html',
                                   result_type='disease',
                                   prediction=prediction,
                                   management=management_info,
                                   image_url=url_for('static', filename=f'uploads/{filename}'))
        else:
            flash('Invalid file type. Please upload PNG, JPG, or JPEG.', 'danger')
            return redirect(url_for('disease_detection'))
    
    return render_template('disease.html')


@app.route('/admin/delete_user/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user (admin only)"""
    try:
        user = User.query.get_or_404(user_id)
        
        # Prevent deleting admin users
        if user.role == 'admin':
            return jsonify({'success': False, 'message': 'Cannot delete admin users'}), 403
        
        # Delete user's predictions first (cascade delete)
        PredictionHistory.query.filter_by(user_id=user_id).delete()
        SoilData.query.filter_by(user_id=user_id).delete()
        
        # Delete the user
        db.session.delete(user)
        db.session.commit()
        
        flash(f'User {user.name} deleted successfully.', 'success')
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/history')
@login_required
def history():
    """View prediction history"""
    predictions = PredictionHistory.query.filter_by(user_id=session['user_id'])\
        .order_by(PredictionHistory.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)


# ==================== Application Setup ====================

def init_db():
    """Initialize database"""
    with app.app_context():
        db.create_all()
        print("[OK] Database initialized successfully")
        
        # Create admin user if not exists
        admin = User.query.filter_by(email='admin@crop.ai').first()
        if not admin:
            admin = User(name='Admin', email='admin@crop.ai', role='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("[OK] Default admin user created (admin@crop.ai / admin123)")


if __name__ == '__main__':
    # Create upload directory if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('database', exist_ok=True)
    
    # Initialize database
    init_db()
    
    # Load ML models
    load_disease_model()
    load_crop_models()
    
    # Run application (FIXED FOR RAILWAY)
    print("\n" + "="*60)
    print("AI-Driven Crop Disease Prediction System")
    print("="*60)
    
    port = int(os.environ.get("PORT", 5000))   # 🔥 IMPORTANT
    print(f"Server starting on port {port}")
    
    app.run(debug=False, host='0.0.0.0', port=port)   # 🔥 FIXED
    
