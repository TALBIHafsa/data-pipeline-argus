import pandas as pd
import joblib
import logging
from typing import Dict, Any, Optional
from extractors.mongo_extractor import MongoExtractor
from transformers.data_transformer import DataTransformer
from utils.config import Config

logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self):
        self.extractor = MongoExtractor()
        self.transformer = DataTransformer()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle ML"""
        try:
            self.model = joblib.load(Config.MODEL_PATH)
            logger.info("Modèle chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
    
    def run_full_pipeline(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Exécute le pipeline complet: extraction, transformation, prédiction
        """
        try:
            # Extraction
            logger.info("Début de l'extraction des données")
            raw_data = self.extractor.extract_all_cars(limit=limit)
            
            if raw_data.empty:
                return {"status": "error", "message": "Aucune donnée extraite"}
            
            logger.info(f"Données extraites: {len(raw_data)} lignes")
            
            # Transformation
            logger.info("Début de la transformation des données")
            transformed_data = self.transformer.transform_mongodb_to_ml_format(raw_data)
            
            if transformed_data.empty:
                return {"status": "error", "message": "Aucune donnée après transformation"}
            
            logger.info(f"Données transformées: {len(transformed_data)} lignes")
            
            # Vérifier si on a assez de données pour les prédictions
            if len(transformed_data) < 1:
                return {"status": "error", "message": "Pas assez de données pour les prédictions"}
            
            # Prédictions seulement si on a les colonnes nécessaires
            predictions = None
            if self._can_generate_predictions(transformed_data):
                logger.info("Début des prédictions")
                predictions = self.generate_predictions(transformed_data)
            else:
                logger.warning("Impossible de générer des prédictions - colonnes manquantes")
            
            # Statistiques
            stats = self._generate_stats(transformed_data, predictions)
            
            return {
                "status": "success",
                "total_processed": len(transformed_data),
                "predictions_generated": len(predictions) if predictions is not None else 0,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def run_incremental_pipeline(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Exécute le pipeline pour les nouvelles données uniquement
        """
        try:
            # Extraction des nouvelles données
            logger.info(f"Extraction des nouvelles données ({hours_back}h)")
            raw_data = self.extractor.extract_new_cars(hours_back=hours_back)
            
            if raw_data.empty:
                return {"status": "info", "message": "Aucune nouvelle donnée"}
            
            # Transformation
            transformed_data = self.transformer.transform_mongodb_to_ml_format(raw_data)
            
            if transformed_data.empty:
                return {"status": "error", "message": "Aucune donnée après transformation"}
            
            # Prédictions
            predictions = None
            if self._can_generate_predictions(transformed_data):
                predictions = self.generate_predictions(transformed_data)
            
            return {
                "status": "success",
                "new_data_processed": len(transformed_data),
                "predictions_generated": len(predictions) if predictions is not None else 0,
                "sample_predictions": predictions[:5].tolist() if predictions is not None and len(predictions) > 0 else []
            }
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline incrémental: {e}")
            return {"status": "error", "message": str(e)}
    
    def _can_generate_predictions(self, data: pd.DataFrame) -> bool:
        """
        Vérifie si on peut générer des prédictions avec les données disponibles
        """
        required_columns = ['city', 'fuel', 'boite_vitesse', 'brand_name', 
                           'model_name', 'model_year', 'mileage', 'first_hand']
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.warning(f"Colonnes manquantes pour les prédictions: {missing_columns}")
            return False
        
        # Vérifier s'il y a assez de données non-nulles
        non_null_counts = data[required_columns].notna().sum()
        insufficient_data = non_null_counts[non_null_counts < len(data) * 0.5]
        
        if not insufficient_data.empty:
            logger.warning(f"Données insuffisantes pour les colonnes: {insufficient_data.index.tolist()}")
            return False
        
        return True
    
    def generate_predictions(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Génère des prédictions pour les données fournies
        """
        if self.model is None:
            logger.error("Modèle non chargé")
            return None
        
        try:
            # Sélectionner les colonnes dans l'ordre attendu par le modèle
            feature_columns = ['city', 'fuel', 'boite_vitesse', 'brand_name', 
                             'model_name', 'model_year', 'mileage', 'first_hand']
            
            # Vérifier que toutes les colonnes sont présentes
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                logger.error(f"Colonnes manquantes: {missing_cols}")
                return None
            
            # Préparer les données pour la prédiction
            X = data[feature_columns].copy()
            
            # Vérifier s'il y a des valeurs manquantes
            if X.isna().any().any():
                logger.warning("Valeurs manquantes détectées, remplissage avec des valeurs par défaut")
                # Remplir les valeurs manquantes avec des valeurs par défaut
                X = X.fillna({
                    'city': 'unknown',
                    'fuel': 'essence',
                    'boite_vitesse': 'manuelle',
                    'brand_name': 'unknown',
                    'model_name': 'unknown',
                    'model_year': 2015,
                    'mileage': 100000,
                    'first_hand': 0
                })
            
            # Générer les prédictions
            predictions = self.model.predict(X)
            
            return pd.Series(predictions, index=data.index)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des prédictions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_stats(self, data: pd.DataFrame, predictions: Optional[pd.Series]) -> Dict[str, Any]:
        """
        Génère des statistiques sur les données et prédictions
        """
        try:
            stats = {
                "data_shape": data.shape,
                "available_columns": list(data.columns),
                "prediction_stats": None,
                "data_stats": {}
            }
            
            # Statistiques sur les prédictions
            if predictions is not None and len(predictions) > 0:
                stats["prediction_stats"] = {
                    "count": len(predictions),
                    "mean": float(predictions.mean()),
                    "median": float(predictions.median()),
                    "min": float(predictions.min()),
                    "max": float(predictions.max()),
                    "std": float(predictions.std())
                }
            
            # Statistiques sur les données
            if 'price' in data.columns:
                price_stats = data['price'].describe()
                stats["data_stats"]["price"] = {
                    "mean": float(price_stats['mean']),
                    "median": float(data['price'].median()),
                    "min": float(price_stats['min']),
                    "max": float(price_stats['max'])
                }
            
            if 'mileage' in data.columns:
                mileage_stats = data['mileage'].describe()
                stats["data_stats"]["mileage"] = {
                    "mean": float(mileage_stats['mean']),
                    "median": float(data['mileage'].median()),
                    "min": float(mileage_stats['min']),
                    "max": float(mileage_stats['max'])
                }
            
            if 'model_year' in data.columns:
                year_stats = data['model_year'].describe()
                stats["data_stats"]["model_year"] = {
                    "mean": float(year_stats['mean']),
                    "median": float(data['model_year'].median()),
                    "min": float(year_stats['min']),
                    "max": float(year_stats['max'])
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des stats: {e}")
            return {"error": str(e)}
    
    def get_db_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur la base de données
        """
        return self.extractor.get_collection_stats()
    
    def close(self):
        """Ferme les connexions"""
        self.extractor.close()
    
    def debug_pipeline(self, limit: int = 5) -> Dict[str, Any]:
        """
        Fonction de debug pour tracer le pipeline étape par étape
        """
        debug_info = {
            "steps": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            # Étape 1: Extraction
            debug_info["steps"].append("1. Extraction des données")
            raw_data = self.extractor.extract_all_cars(limit=limit)
            
            if raw_data.empty:
                debug_info["errors"].append("Aucune donnée extraite")
                return debug_info
            
            debug_info["steps"].append(f"   -> {len(raw_data)} lignes extraites")
            debug_info["steps"].append(f"   -> Colonnes: {list(raw_data.columns)}")
            
            # Étape 2: Transformation
            debug_info["steps"].append("2. Transformation des données")
            transformed_data = self.transformer.transform_mongodb_to_ml_format(raw_data)
            
            if transformed_data.empty:
                debug_info["errors"].append("Aucune donnée après transformation")
                return debug_info
            
            debug_info["steps"].append(f"   -> {len(transformed_data)} lignes transformées")
            debug_info["steps"].append(f"   -> Colonnes: {list(transformed_data.columns)}")
            
            # Étape 3: Vérification des colonnes pour prédictions
            debug_info["steps"].append("3. Vérification des colonnes pour prédictions")
            can_predict = self._can_generate_predictions(transformed_data)
            debug_info["steps"].append(f"   -> Peut prédire: {can_predict}")
            
            if can_predict:
                debug_info["steps"].append("4. Génération des prédictions")
                predictions = self.generate_predictions(transformed_data)
                if predictions is not None:
                    debug_info["steps"].append(f"   -> {len(predictions)} prédictions générées")
                else:
                    debug_info["errors"].append("Échec de la génération des prédictions")
            else:
                debug_info["warnings"].append("Prédictions impossibles - colonnes manquantes")
            
            # Échantillon de données
            debug_info["sample_data"] = transformed_data.head(3).to_dict()
            
            return debug_info
            
        except Exception as e:
            debug_info["errors"].append(f"Erreur dans le debug: {str(e)}")
            return debug_info