from fastapi import FastAPI, Header, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import asyncio
from contextlib import asynccontextmanager
import uvicorn
import traceback
from pydantic import ValidationError

# Imports des modules du pipeline
from api.schemas import CarFeatureInput, PredictionResponse, PipelineResponse, DatabaseStats, Settings
from pipelines.etl_pipeline import ETLPipeline
from utils.config import Config

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variables globales
pipeline = None
model = None
settings = Settings()
DEBUG = settings.debug

# Gestionnaire de contexte pour l'initialisation
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    global pipeline, model
    
    # Initialisation au démarrage
    logger.info("Initialisation de l'application...")
    
    try:
        # Chargement du modèle
        import joblib
        model = joblib.load(Config.MODEL_PATH)
        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        model = None
    
    try:
        # Initialisation du pipeline
        pipeline = ETLPipeline()
        logger.info("Pipeline initialisé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du pipeline: {e}")
        pipeline = None
    
    yield
    
    # Nettoyage à la fermeture
    logger.info("Fermeture de l'application...")
    if pipeline:
        try:
            pipeline.close()
            logger.info("Pipeline fermé proprement")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture du pipeline: {e}")

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Argus Prediction API with Data Pipeline",
    description="API pour prédire les prix des voitures avec pipeline de données automatisé",
    version="2.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sécurité API
API_KEY = settings.api_key
security = HTTPBearer()

def verify_key(x_api_key: str = Header(...)):
    """Vérifie la clé API dans les headers"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérifie le token Bearer"""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Token invalide")

def get_pipeline():
    """Retourne l'instance du pipeline"""
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline non initialisé")
    return pipeline

def get_model():
    """Retourne l'instance du modèle"""
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    return model

# === ENDPOINTS DE BASE ===

@app.get("/")
def root():
    """Endpoint racine avec informations sur l'API"""
    return {
        "message": "Welcome to the Argus Prediction API with Data Pipeline",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "predictions": "/predict",
            "pipeline": {
                "full": "/pipeline/run",
                "incremental": "/pipeline/run/incremental", 
                "status": "/pipeline/status"
            },
            "database": {
                "stats": "/database/stats",
                "debug": "/database/debug"
            },
            "monitoring": {
                "metrics": "/metrics",
                "logs": "/logs"
            }
        },
        "authentication": "API Key required in X-API-Key header"
    }

@app.get("/health")
def health_check():
    """Vérification de l'état de santé de l'API"""
    global pipeline, model
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "healthy",
            "model": "healthy" if model is not None else "unhealthy",
            "pipeline": "healthy" if pipeline is not None else "unhealthy"
        }
    }
    
    # Vérifier la connexion à la base de données
    try:
        if pipeline:
            db_stats = pipeline.get_db_stats()
            health_status["components"]["database"] = "healthy"
            health_status["database_info"] = db_stats
        else:
            health_status["components"]["database"] = "unhealthy"
    except Exception as e:
        health_status["components"]["database"] = "unhealthy"
        health_status["database_error"] = str(e)
    
    # Déterminer le statut global
    if any(status == "unhealthy" for status in health_status["components"].values()):
        health_status["status"] = "degraded"
    
    return health_status

# === ENDPOINTS DE PRÉDICTION ===

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_key)])
def predict(input_data: CarFeatureInput):
    """
    Prédire le prix d'une voiture à partir de ses caractéristiques
    """
    try:
        model = get_model()
        
        # Convertir les données d'entrée en DataFrame
        features_df = pd.DataFrame([input_data.dict()])
        
        # Faire la prédiction
        prediction = model.predict(features_df)[0]
        
        # Créer la réponse
        response = PredictionResponse(
            predicted_argus=float(prediction),
            input_data=input_data,
            timestamp=datetime.now()
        )
        
        logger.info(f"Prédiction effectuée: {prediction:.2f} pour {input_data.brand_name} {input_data.model_name}")
        
        return response
        
    except ValidationError as e:
        logger.error(f"Erreur de validation: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/predict/batch", dependencies=[Depends(verify_key)])
def predict_batch(input_data: List[CarFeatureInput]):
    """
    Prédire les prix pour plusieurs voitures
    """
    try:
        model = get_model()
        
        if len(input_data) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 prédictions par batch")
        
        # Convertir en DataFrame
        features_df = pd.DataFrame([item.dict() for item in input_data])
        
        # Faire les prédictions
        predictions = model.predict(features_df)
        
        # Créer les réponses
        responses = []
        for i, (pred, input_item) in enumerate(zip(predictions, input_data)):
            response = PredictionResponse(
                predicted_argus=float(pred),
                input_data=input_item,
                timestamp=datetime.now()
            )
            responses.append(response)
        
        logger.info(f"Prédictions batch effectuées: {len(responses)} éléments")
        
        return {
            "predictions": responses,
            "total_predictions": len(responses),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction batch: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

# === ENDPOINTS DU PIPELINE ===

@app.post("/pipeline/run", response_model=PipelineResponse, dependencies=[Depends(verify_key)])
def run_full_pipeline(
    background_tasks: BackgroundTasks,
    limit: Optional[int] = Query(None, description="Limite du nombre d'enregistrements à traiter")
):
    """
    Exécute le pipeline complet de traitement des données
    """
    try:
        pipeline = get_pipeline()
        
        logger.info(f"Démarrage du pipeline complet avec limit={limit}")
        
        # Exécuter le pipeline
        result = pipeline.run_full_pipeline(limit=limit)
        
        # Créer la réponse
        response = PipelineResponse(
            status=result.get("status", "unknown"),
            message=result.get("message"),
            total_processed=result.get("total_processed"),
            predictions_generated=result.get("predictions_generated"),
            stats=result.get("stats"),
            sample_predictions=result.get("sample_predictions", [])[:10]  # Limiter à 10 échantillons
        )
        
        logger.info(f"Pipeline terminé: {response.status}")
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/pipeline/run/incremental", response_model=PipelineResponse, dependencies=[Depends(verify_key)])
def run_incremental_pipeline(
    hours_back: int = Query(24, description="Nombre d'heures en arrière pour les nouvelles données")
):
    """
    Exécute le pipeline pour les nouvelles données uniquement
    """
    try:
        pipeline = get_pipeline()
        
        logger.info(f"Démarrage du pipeline incrémental pour {hours_back}h")
        
        # Exécuter le pipeline incrémental
        result = pipeline.run_incremental_pipeline(hours_back=hours_back)
        
        # Créer la réponse
        response = PipelineResponse(
            status=result.get("status", "unknown"),
            message=result.get("message"),
            total_processed=result.get("new_data_processed"),
            predictions_generated=result.get("predictions_generated"),
            sample_predictions=result.get("sample_predictions", [])
        )
        
        logger.info(f"Pipeline incrémental terminé: {response.status}")
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du pipeline incrémental: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/pipeline/status", dependencies=[Depends(verify_key)])
def get_pipeline_status():
    """
    Obtient le statut actuel du pipeline
    """
    try:
        pipeline = get_pipeline()
        
        # Obtenir les statistiques de la base de données
        db_stats = pipeline.get_db_stats()
        
        return {
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "database_stats": db_stats,
            "pipeline_components": {
                "extractor": "active",
                "transformer": "active",
                "model": "loaded" if model is not None else "not_loaded"
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/pipeline/debug", dependencies=[Depends(verify_key)])
def debug_pipeline(
    limit: int = Query(5, description="Nombre d'enregistrements pour le debug")
):
    """
    Debug du pipeline étape par étape
    """
    try:
        pipeline = get_pipeline()
        
        logger.info(f"Démarrage du debug pipeline avec limit={limit}")
        
        # Exécuter le debug
        debug_result = pipeline.debug_pipeline(limit=limit)
        
        return {
            "debug_info": debug_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du debug: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

# === ENDPOINTS DE BASE DE DONNÉES ===

@app.get("/database/stats", response_model=DatabaseStats, dependencies=[Depends(verify_key)])
def get_database_stats():
    """
    Obtient les statistiques de la base de données
    """
    try:
        pipeline = get_pipeline()
        
        stats = pipeline.get_db_stats()
        
        return DatabaseStats(
            total_documents=stats.get("total_documents", 0),
            documents_with_price=stats.get("documents_with_price", 0),
            latest_date=stats.get("latest_date"),
            collection_name=stats.get("collection_name", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/database/debug", dependencies=[Depends(verify_key)])
def debug_database(
    limit: int = Query(3, description="Nombre d'enregistrements pour le debug")
):
    """
    Debug de l'extraction de données
    """
    try:
        pipeline = get_pipeline()
        
        # Utiliser la fonction de debug de l'extractor
        debug_info = pipeline.extractor.debug_data_pipeline(limit=limit)
        
        return {
            "debug_info": debug_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du debug database: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

# === ENDPOINTS DE MONITORING ===

@app.get("/metrics", dependencies=[Depends(verify_key)])
def get_metrics():
    """
    Obtient les métriques de l'API
    """
    try:
        pipeline = get_pipeline()
        db_stats = pipeline.get_db_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "database_metrics": db_stats,
            "api_status": {
                "model_loaded": model is not None,
                "pipeline_active": pipeline is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/logs", dependencies=[Depends(verify_key)])
def get_recent_logs(
    lines: int = Query(50, description="Nombre de lignes de log à retourner")
):
    """
    Retourne les logs récents (implémentation basique)
    """
    try:
        # Cette implémentation basique retourne des informations de statut
        # Pour une implémentation complète, vous devriez lire un fichier de log
        return {
            "timestamp": datetime.now().isoformat(),
            "message": "Logs endpoint - implémentation basique",
            "status": "Cette fonctionnalité nécessite la configuration d'un système de logging centralisé"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des logs: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

# === GESTION DES ERREURS ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Gestionnaire d'erreurs HTTP personnalisé"""
    logger.error(f"HTTP Error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gestionnaire d'erreurs générales"""
    logger.error(f"Erreur non gérée: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du serveur",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# === FONCTION MAIN ===




def main():
    """Fonction principale pour lancer l'API"""
    logger.info("Démarrage de l'API Argus Prediction...")
    
    # Configuration du serveur
    config = {
        "host": "127.0.0.1",  # Changé de 0.0.0.0 à 127.0.0.1 pour dev local
        "port": 8000,
        "log_level": "info" if not DEBUG else "debug",
        "reload": DEBUG
    }
    
    logger.info(f"Configuration serveur: {config}")
    logger.info(f"API accessible sur: http://127.0.0.1:8000/")
    
    # Lancement du serveur
    uvicorn.run(
        "enhanced_api:app",
        **config
    )


if __name__ == "__main__":
    main()