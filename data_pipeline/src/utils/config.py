import os
from typing import Any, Dict, Optional

class Config:
    """Configuration centralisée pour l'application"""
    
    # Configuration MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "car_database")
    MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "cars")
    
    # Configuration du modèle ML
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/argus_model.joblib")
    
    # Configuration API
    API_KEY: str = os.getenv("API_KEY", "api-key")
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Configuration de sécurité
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Configuration du logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "logs/api.log")
    
    # Configuration des limites
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "100"))
    MAX_PIPELINE_RECORDS: int = int(os.getenv("MAX_PIPELINE_RECORDS", "10000"))
    
    # Configuration du cache
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    
    # Configuration des métriques
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8001"))
    
    # Configuration de debug
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    TESTING: bool = os.getenv("TESTING", "false").lower() == "true"
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Valide la configuration et retourne les erreurs"""
        errors = {}
        
        # Vérifier que le modèle existe
        if not os.path.exists(cls.MODEL_PATH):
            errors["MODEL_PATH"] = f"Le fichier modèle n'existe pas: {cls.MODEL_PATH}"
        
        # Vérifier la clé API
        if cls.API_KEY == "api-key":
            errors["API_KEY"] = "La clé API par défaut est utilisée. Changez-la pour la production."
        
        # Vérifier la clé secrète
        if cls.SECRET_KEY == "your-secret-key-here":
            errors["SECRET_KEY"] = "La clé secrète par défaut est utilisée. Changez-la pour la production."
        
        # Vérifier la configuration MongoDB
        if "localhost" in cls.MONGODB_URI and not cls.DEBUG:
            errors["MONGODB_URI"] = "Configuration MongoDB en localhost détectée en production."
        
        return errors
    
    @classmethod
    def get_database_config(cls) -> Dict[str, str]:
        """Retourne la configuration de la base de données"""
        return {
            "uri": cls.MONGODB_URI,
            "database": cls.MONGODB_DATABASE,
            "collection": cls.MONGODB_COLLECTION
        }
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Retourne la configuration de l'API"""
        return {
            "host": cls.API_HOST,
            "port": cls.API_PORT,
            "debug": cls.DEBUG,
            "max_batch_size": cls.MAX_BATCH_SIZE,
            "max_pipeline_records": cls.MAX_PIPELINE_RECORDS
        }