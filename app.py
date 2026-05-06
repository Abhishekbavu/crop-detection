"""
AI-Driven Crop Disease Prediction and Management System
Main Flask Application (Railway Ready)
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

# ==================== App Initialization ====================

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)

# ==================== Global Variables ====================

disease_model = None
crop_models = {}
model_metrics = {}
scaler = None


# ==================== Database Models ====================

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='farmer')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    soil_data = db.relationship('SoilData', backref='user', lazy=True)
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


class SoilData(db.Model):
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
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)
    result = db.Column(db.String(500), nullable=False)
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ==================== Helper Functions ====================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if user.role != 'admin':
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ==================== Model Loading ====================

def load_disease_model():
    global disease_model
    try:
        import tensorflow as tf
        model_path = app.config['MODEL_PATH']
        if os.path.exists(model_path):
            disease_model = tf.keras.models.load_model(model_path)
            print("✓ Disease model loaded")
        else:
            print("⚠ Disease model not found")
    except Exception as e:
        print(f"Error loading disease model: {e}")


def load_crop_models():
    global crop_models, model_metrics, scaler

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, 'models', 'crop_models')

    if not os.path.exists(models_dir):
        print("⚠ Crop models not found")
        return

    try:
        files = {
            'Decision Tree': 'decision_tree_model.pkl',
            'Random Forest': 'random_forest_model.pkl',
            'Logistic Regression': 'logistic_regression_model.pkl'
        }

        for name, file in files.items():
            path = os.path.join(models_dir, file)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    crop_models[name] = pickle.load(f)

        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        metrics_path = os.path.join(models_dir, 'model_metrics.pkl')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                model_metrics = pickle.load(f)

        print("✓ Crop models loaded")

    except Exception as e:
        print(f"Error loading crop models: {e}")


# ==================== INIT ON START ====================

with app.app_context():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    db.create_all()
    load_disease_model()
    load_crop_models()

    admin = User.query.filter_by(email='admin@crop.ai').first()
    if not admin:
        admin = User(name='Admin', email='admin@crop.ai', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()


# ==================== ROUTES ====================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = User(
            name=request.form['name'],
            email=request.form['email']
        )
        user.set_password(request.form['password'])
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and user.check_password(request.form['password']):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        flash("Invalid credentials", "danger")
    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/disease', methods=['GET', 'POST'])
@login_required
def disease_detection():
    if request.method == 'POST':
        file = request.files['plant_image']

        if not file or file.filename == '':
            flash("No file selected", "warning")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file type", "danger")
            return redirect(request.url)

        if disease_model is None:
            return "Model not loaded", 500

        filename = str(datetime.utcnow().timestamp()) + "_" + secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        result = predict_disease(path)
        return render_template('result.html', prediction=result)

    return render_template('disease.html')


# ==================== ENTRY ====================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
