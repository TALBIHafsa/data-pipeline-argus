# Car Price Prediction Data Pipeline & API

A comprehensive data pipeline and RESTful API for predicting car prices using machine learning. The system extracts data from MongoDB, processes it through an ETL pipeline, and provides real-time predictions via FastAPI.



## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Pipeline Architecture](#pipeline-architecture)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- MongoDB 4.0+
- pip or conda

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd car-price-prediction-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up MongoDB**
```bash
# Make sure MongoDB is running
mongod --dbpath /path/to/your/db
```

5. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=car_database
MONGODB_COLLECTION=cars

# Model Configuration
MODEL_PATH=models/argus_model.joblib

# API Configuration
API_KEY=api-key
API_HOST=0.0.0.0
API_PORT=8000

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# Limits
MAX_BATCH_SIZE=100
MAX_PIPELINE_RECORDS=10000

# Debug
DEBUG=false
TESTING=false
```

### Required Model

Place your trained model file at the specified `MODEL_PATH`. The model should be a joblib-serialized scikit-learn model expecting these features:

- `city`: Car location
- `fuel`: Fuel type
- `boite_vitesse`: Transmission type
- `brand_name`: Car brand
- `model_name`: Car model
- `model_year`: Manufacturing year
- `mileage`: Vehicle mileage
- `first_hand`: First-hand ownership (0/1)

## 🚀 Usage

### Starting the API

```bash
python enhanced_api.py
```

The API will be available at `http://localhost:8000`

### Basic API Usage

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "casablanca",
    "fuel": "essence",
    "boite_vitesse": "automatique",
    "brand_name": "toyota",
    "model_name": "corolla",
    "model_year": 2020,
    "mileage": 50000,
    "first_hand": 1
  }'
```

#### Run Full Pipeline

```bash
curl -X POST "http://localhost:8000/pipeline/run" \
  -H "X-API-Key: your-api-key"
```

## 📚 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and available endpoints |
| `GET` | `/health` | Health check with component status |

### Prediction Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single car price prediction |
| `POST` | `/predict/batch` | Batch predictions (max 100) |

### Pipeline Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/pipeline/run` | Run complete ETL pipeline |
| `POST` | `/pipeline/run/incremental` | Process new data only |
| `GET` | `/pipeline/status` | Get pipeline status |
| `POST` | `/pipeline/debug` | Debug pipeline execution |

### Database Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/database/stats` | Database statistics |
| `GET` | `/database/debug` | Debug database extraction |

### Monitoring Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/metrics` | System metrics |
| `GET` | `/logs` | Recent logs |

## 🏗️ Pipeline Architecture

### ETL Pipeline Flow

```
MongoDB → Data Extraction → Data Transformation → ML Prediction → Results
```

### Components

1. **MongoExtractor**: Extracts car data from MongoDB
2. **DataTransformer**: Cleans and transforms data for ML model
3. **ETLPipeline**: Orchestrates the entire pipeline
4. **ML Model**: Generates price predictions

### Data Transformation Steps

1. **Field Mapping**: Java to Python field conversion
2. **Data Extraction**: Extract values from MongoDB dictionaries
3. **Data Cleaning**: Remove invalid records and normalize values
4. **Feature Engineering**: Create mileage from min/max values
5. **Filtering**: Apply business rules (price ranges, valid years)
6. **Validation**: Ensure data quality for predictions

## 🔧 Development

### Project Structure

```
├── api/
│   ├── schemas.py          # Pydantic models
│   └── __init__.py
├── pipelines/
│   ├── etl_pipeline.py     # Main ETL pipeline
│   └── __init__.py
├── extractors/
│   ├── mongo_extractor.py  # MongoDB data extraction
│   └── __init__.py
├── transformers/
│   ├── data_transformer.py # Data transformation logic
│   └── __init__.py
├── utils/
│   ├── config.py          # Configuration management
│   └── __init__.py
├── models/
│   └── rf_pipeline.pkl # Trained ML model
├── enhanced_api.py       # Main API application
├── requirements.txt      # Python dependencies
├── .env         # Environment configuration template
└── README.md           # This file
```




## 🔍 Troubleshooting

### Common Issues

#### 1. Model Not Loading
```
Error: "Modèle non chargé"
```
**Solution**: Ensure the model file exists at the specified `MODEL_PATH`

#### 2. MongoDB Connection Error
```
Error: "Pipeline non initialisé"
```
**Solution**: Check MongoDB connection and ensure the service is running

#### 3. API Authentication Error
```
Error: "Clé API invalide"
```
**Solution**: Verify the `X-API-Key` header matches your configured API key

#### 4. Data Transformation Issues
```
Error: "Aucune donnée après transformation"
```
**Solution**: Check data quality and ensure required fields are present

### Debug Mode

Enable debug mode for detailed logging:
```bash
DEBUG=true python enhanced_api.py
```

### Health Check

Check system health:
```bash
curl http://localhost:8000/health
```


## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the logs for error details
- Use the debug endpoints for pipeline issues

---

**Note**: This README assumes you have the necessary trained model and MongoDB setup. Adjust the configuration according to your specific environment and requirements.
