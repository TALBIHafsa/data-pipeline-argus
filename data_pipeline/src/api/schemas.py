from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class CarFeatureInput(BaseModel):
    city: str = Field(..., description="Ville de la voiture")
    fuel: str = Field(..., description="Type de carburant")
    boite_vitesse: str = Field(..., description="Type de boîte de vitesse")
    brand_name: str = Field(..., description="Marque de la voiture")
    model_name: str = Field(..., description="Modèle de la voiture")
    model_year: int = Field(..., ge=1990, le=2025, description="Année du modèle")
    mileage: float = Field(..., ge=1, le=600000, description="Kilométrage")
    first_hand: int = Field(..., ge=0, le=1, description="Première main (0=non, 1=oui)")

class PredictionResponse(BaseModel):
    predicted_argus: float = Field(..., description="Prix prédit par le modèle")
    input_data: CarFeatureInput = Field(..., description="Données d'entrée utilisées")
    timestamp: datetime = Field(default_factory=datetime.now)

class PipelineResponse(BaseModel):
    status: str = Field(..., description="Statut de l'exécution")
    message: Optional[str] = Field(None, description="Message d'information")
    total_processed: Optional[int] = Field(None, description="Nombre total de données traitées")
    predictions_generated: Optional[int] = Field(None, description="Nombre de prédictions générées")
    stats: Optional[Dict[str, Any]] = Field(None, description="Statistiques sur les données")
    sample_predictions: Optional[List[float]] = Field(None, description="Échantillon de prédictions")

class DatabaseStats(BaseModel):
    total_documents: int
    documents_with_price: int
    latest_date: Optional[str]
    collection_name: str

class Settings(BaseModel):
    debug: bool = Field(default=False)
    api_key: str = Field(default="your_api_key_here")
    mongodb_uri: str = Field(default="mongodb://localhost:27017/")
    mongodb_database: str = Field(default="your_database_name")
    mongodb_collection: str = Field(default="used_cars")