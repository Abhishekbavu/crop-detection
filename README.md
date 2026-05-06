# AI-Driven Crop Disease Prediction and Management System

A comprehensive Flask-based web application that provides AI-powered agricultural solutions for farmers and agricultural experts.

## Features

- 🔐 **User Authentication** - Secure login/registration with role-based access (Farmer/Admin)
- 🌾 **Crop Recommendation** - AI-based crop suggestions based on soil NPK values and weather conditions
- 🧪 **Fertilizer Suggestion** - Smart fertilizer recommendations with dosage amounts
- 🔬 **Plant Disease Detection** - Image-based disease identification using deep learning
- 📊 **Dashboard** - Comprehensive metrics and prediction history
- 💾 **SQLite Database** - Persistent data storage with SQLAlchemy ORM

## Tech Stack

- **Backend**: Flask 3.0, SQLAlchemy
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Database**: SQLite
- **ML**: TensorFlow/Keras (.h5 model)
- **Security**: Bcrypt password hashing

## Project Structure

```
crop-detection/
│
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── create_model.py             # Script to generate ML model
├── requirements.txt            # Python dependencies
│
├── models/
│   └── disease_model.h5        # TensorFlow disease detection model
│
├── database/
│   └── app.db                  # SQLite database (auto-created)
│
├── static/
│   ├── uploads/                # Uploaded plant images
│   ├── js/                     # JavaScript files
│   └── images/                 # Static images
│
└── templates/
    ├── base.html               # Base template
    ├── login.html              # Login page
    ├── register.html           # Registration page
    ├── dashboard.html          # User dashboard
    ├── crop.html               # Crop recommendation form
    ├── fertilizer.html         # Fertilizer recommendation form
    ├── disease.html            # Disease detection upload
    └── result.html             # Results display
```

## Installation & Setup

### 1. Clone or Download the Project

```bash
cd crop-detection
```

### 2. Create Virtual Environment (Recommended)

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Or using Command Prompt
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create the ML Model

```powershell
python create_model.py
```

This will generate a dummy `disease_model.h5` file in the `models/` directory. For production use, train the model with an actual plant disease dataset like PlantVillage.

### 5. Run the Application

```powershell
python app.py
```

The application will:
- Create the database automatically
- Create a default admin user
- Load the ML model
- Start the server at `http://127.0.0.1:5000`

## Default Credentials

**Admin Account:**
- Email: `admin@crop.ai`
- Password: `admin123`

## Usage Guide

### 1. User Registration
- Navigate to the register page
- Fill in your details (name, email, password)
- Select role (Farmer or Admin)
- Click "Create Account"

### 2. Crop Recommendation
- Login to your account
- Go to "Crop Recommendation"
- Enter soil NPK values (Nitrogen, Phosphorus, Potassium)
- Enter weather conditions (Temperature, Humidity, Rainfall)
- Click "Get Crop Recommendations"
- View top 3 recommended crops with match percentages

### 3. Fertilizer Suggestion
- Go to "Fertilizer" page
- Select your crop
- Enter current soil NPK values
- Click "Get Fertilizer Recommendations"
- View nutrient analysis and fertilizer dosage recommendations

### 4. Disease Detection
- Go to "Disease Detection"
- Upload a clear image of the plant leaf
- Click "Analyze Plant Image"
- View detection results with:
  - Disease name
  - Confidence score
  - Risk level
  - Prevention measures
  - Treatment recommendations

### 5. Dashboard
- View your prediction statistics
- Access recent prediction history
- Quick access to all features

## Database Models

### User
- `id`: Primary key
- `name`: Full name
- `email`: Unique email address
- `password`: Hashed password (bcrypt)
- `role`: farmer or admin
- `created_at`: Registration timestamp

### SoilData
- `id`: Primary key
- `user_id`: Foreign key to User
- `nitrogen, phosphorus, potassium`: NPK values
- `temperature, humidity, rainfall`: Weather data
- `created_at`: Entry timestamp

### PredictionHistory
- `id`: Primary key
- `user_id`: Foreign key to User
- `prediction_type`: crop/fertilizer/disease
- `result`: Prediction result text
- `confidence`: Confidence score (%)
- `created_at`: Prediction timestamp

## ML Model Information

The system uses a Convolutional Neural Network (CNN) for disease detection:
- **Input**: 224x224 RGB images
- **Output**: 12 disease classes (configurable in `config.py`)
- **Format**: TensorFlow/Keras .h5 file

### Training Your Own Model

For production use, train the model with actual datasets:

1. **Recommended Datasets:**
   - PlantVillage Dataset
   - PlantDoc Dataset
   - Kaggle Plant Disease datasets

2. **Training Steps:**
   - Collect and preprocess images (224x224)
   - Split into train/validation/test sets
   - Train using the architecture in `create_model.py`
   - Save trained model as `disease_model.h5`

## Security Features

- ✅ Password hashing with bcrypt
- ✅ Session-based authentication
- ✅ File upload validation (type and size)
- ✅ Secure filename handling
- ✅ Role-based access control
- ✅ SQL injection prevention (SQLAlchemy ORM)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page (redirects to login) |
| `/register` | GET, POST | User registration |
| `/login` | GET, POST | User login |
| `/logout` | GET | User logout |
| `/dashboard` | GET | User dashboard |
| `/crop` | GET, POST | Crop recommendation |
| `/fertilizer` | GET, POST | Fertilizer suggestion |
| `/disease` | GET, POST | Disease detection |

## Configuration

Edit `config.py` to customize:
- Database URI
- Upload folder location
- Max file size
- Allowed file extensions
- ML model path
- Crop list
- Disease classes
- Fertilizer types
- Disease management data

## Troubleshooting

### Model not loading
```
⚠ Warning: Model file not found at models/disease_model.h5
```
**Solution:** Run `python create_model.py` to create the model file.

### Database errors
**Solution:** Delete `database/app.db` and restart the application to recreate the database.

### Port already in use
**Solution:** Change the port in `app.py`:
```python
app.run(debug=True, host='127.0.0.1', port=5001)
```

### Module not found errors
**Solution:** Ensure all dependencies are installed:
```powershell
pip install -r requirements.txt
```

## Future Enhancements

- 📱 Mobile responsive improvements
- 🌍 Multi-language support
- 📈 Advanced analytics dashboard
- 🗺️ Geographic crop suitability mapping
- 🤖 Chatbot for agricultural advice
- 📧 Email notifications for critical alerts
- 📊 Export data as PDF/CSV
- 🔄 Real-time weather API integration

## License

This is an academic project for educational purposes.

## Support

For issues, questions, or contributions, please create an issue in the project repository.

---

**Built with ❤️ for farmers and agricultural innovation**
