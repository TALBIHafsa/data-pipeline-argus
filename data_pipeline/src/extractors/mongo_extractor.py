import pymongo
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any, List
import numpy as np
from utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoExtractor:
    def __init__(self):
        self.client = pymongo.MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.MONGODB_DATABASE]
        self.collection = self.db[Config.MONGODB_COLLECTION]
    
    def _extract_value_from_dict(self, field_value: Any, default: str = 'unknown') -> str:
        """
        Extrait la valeur d'un champ complexe (dictionnaire) MongoDB
        Priorité: 'label' > 'translations.fr' > 'translations.ar' > 'value'
        """
        if pd.isna(field_value) or field_value is None:
            return default
        
        if isinstance(field_value, dict):
            # Debug: afficher la structure pour comprendre le problème
            logger.debug(f"Processing dict: {field_value}")
            
            # Priorité 1: 'label' (souvent c'est le nom lisible)
            if 'label' in field_value and field_value['label']:
                value = str(field_value['label']).strip().lower()
                if value and value != 'null' and value != 'none':
                    return value
            
            # Priorité 2: 'translations.fr' (traduction française)
            if 'translations' in field_value and isinstance(field_value['translations'], dict):
                if 'fr' in field_value['translations'] and field_value['translations']['fr']:
                    value = str(field_value['translations']['fr']).strip().lower()
                    if value and value != 'null' and value != 'none':
                        return value
                elif 'ar' in field_value['translations'] and field_value['translations']['ar']:
                    value = str(field_value['translations']['ar']).strip()
                    if value and value != 'null' and value != 'none':
                        return value
            
            # Priorité 3: 'value' (en dernier car peut être un ID)
            if 'value' in field_value and field_value['value']:
                value = str(field_value['value']).strip().lower()
                if value and value != 'null' and value != 'none':
                    # Si c'est un nombre pur (comme '5'), on évite de l'utiliser
                    if not value.isdigit():
                        return value
        
        # Si c'est déjà une chaîne simple
        if isinstance(field_value, str):
            value = field_value.strip().lower()
            if value and value != 'null' and value != 'none':
                return value
        
        return default
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et valide les données transformées avec des logs détaillés
        """
        df = df.copy()
        initial_count = len(df)
        logger.info(f"Starting data cleaning with {initial_count} records")
        
        # Nettoyer les prix (supprimer les valeurs aberrantes)
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            price_null_count = df['price'].isna().sum()
            logger.info(f"Price conversion: {price_null_count} null values after conversion")
            
            if not df.empty:
                price_stats = df['price'].describe()
                logger.info(f"Price statistics: min={price_stats['min']}, max={price_stats['max']}, mean={price_stats['mean']:.2f}")
            
            # Ajuster les seuils pour être plus permissifs
            before_price_filter = len(df)
            df = df[(df['price'] > 500) & (df['price'] < 50000000)]  # Seuils plus larges
            after_price_filter = len(df)
            logger.info(f"Price filtering: {before_price_filter} -> {after_price_filter} records")
        
        # Nettoyer les années
        if 'model_year' in df.columns:
            df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')
            current_year = datetime.now().year
            before_year_filter = len(df)
            df = df[(df['model_year'] >= 1960) & (df['model_year'] <= current_year + 2)]  # Plus permissif
            after_year_filter = len(df)
            logger.info(f"Year filtering: {before_year_filter} -> {after_year_filter} records")
        
        # Nettoyer le kilométrage
        if 'mileage' in df.columns:
            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
            before_mileage_filter = len(df)
            df = df[(df['mileage'] >= 0) & (df['mileage'] <= 2000000)]  # Plus permissif
            after_mileage_filter = len(df)
            logger.info(f"Mileage filtering: {before_mileage_filter} -> {after_mileage_filter} records")
        
        # Nettoyer les champs texte
        text_columns = ['city', 'fuel', 'boite_vitesse', 'first_hand', 'brand_name', 'model_name']
        for col in text_columns:
            if col in df.columns:
                # Remplacer les valeurs vides par 'unknown'
                df[col] = df[col].replace(['', 'null', 'none', np.nan], 'unknown')
                # Standardiser les valeurs
                df[col] = df[col].astype(str).str.strip().str.lower()
                # Remplacer les valeurs 'unknown' par défaut intelligente si possible
                if col == 'fuel' and (df[col] == 'unknown').sum() > len(df) * 0.8:
                    df[col] = df[col].replace('unknown', 'essence')  # Valeur par défaut pour les carburants manquants
        
        final_count = len(df)
        logger.info(f"Data cleaning completed: {initial_count} -> {final_count} records ({final_count/initial_count*100:.1f}% retained)")
        
        return df
    
    def _transform_complex_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme les champs complexes en valeurs simples avec des logs détaillés
        """
        df = df.copy()
        initial_count = len(df)
        logger.info(f"Starting field transformation with {initial_count} records")
        
        # Transformation des champs complexes
        complex_fields = {
            'carCity': 'city',
            'carFuel': 'fuel',
            'carBV': 'boite_vitesse',
            'carFirstHand': 'first_hand',
            'carBrand': 'brand_name',
            'carModel': 'model_name'
        }
        
        for mongo_field, new_field in complex_fields.items():
            if mongo_field in df.columns:
                logger.debug(f"Transforming {mongo_field} -> {new_field}")
                df[new_field] = df[mongo_field].apply(
                    lambda x: self._extract_value_from_dict(x, default='unknown')
                )
                # Log quelques exemples
                sample_values = df[new_field].value_counts().head(3)
                logger.debug(f"{new_field} top values: {sample_values.to_dict()}")
            else:
                # Si le champ n'existe pas, créer une colonne avec 'unknown'
                logger.debug(f"Field {mongo_field} not found, creating {new_field} with 'unknown'")
                df[new_field] = 'unknown'
        
        # Champs simples à renommer
        simple_fields = {
            'carPrice': 'price',
            'carYear': 'model_year',
            'carMileageMin': 'mileage_min',
            'carMileageMax': 'mileage_max'
        }
        
        for mongo_field, new_field in simple_fields.items():
            if mongo_field in df.columns:
                if mongo_field == 'carPrice':
                    # Extraire la valeur numérique du champ price (peut être un dict ou un nombre direct)
                    def extract_price(x):
                        if x is None:
                            return None
                        if isinstance(x, dict) and 'value' in x and x.get('value') not in [None, '']:
                            return float(x.get('value'))
                        elif isinstance(x, (int, float)):
                            return float(x)
                        elif isinstance(x, str) and x.isdigit():
                            return float(x)
                        else:
                            return None
                    
                    df[new_field] = df[mongo_field].apply(extract_price)
                    # Log des statistiques de prix
                    valid_prices = df[new_field].dropna()
                    if not valid_prices.empty:
                        logger.info(f"Price extraction: {len(valid_prices)} valid prices, range: {valid_prices.min():.0f} - {valid_prices.max():.0f}")
                else:
                    df[new_field] = df[mongo_field]
        
        # Créer le champ mileage (moyenne des min et max)
        if 'mileage_min' in df.columns and 'mileage_max' in df.columns:
            mileage_min = pd.to_numeric(df['mileage_min'], errors='coerce')
            mileage_max = pd.to_numeric(df['mileage_max'], errors='coerce')
            
            # Utiliser la moyenne, ou la valeur disponible si une seule existe
            df['mileage'] = np.where(
                pd.isna(mileage_min) & pd.isna(mileage_max),
                np.nan,
                np.where(
                    pd.isna(mileage_min),
                    mileage_max,
                    np.where(
                        pd.isna(mileage_max),
                        mileage_min,
                        (mileage_min + mileage_max) / 2
                    )
                )
            )
            
            valid_mileage = df['mileage'].dropna()
            if not valid_mileage.empty:
                logger.info(f"Mileage calculation: {len(valid_mileage)} valid values, range: {valid_mileage.min():.0f} - {valid_mileage.max():.0f}")
        
        # Nettoyer et valider les données
        df = self._clean_and_validate_data(df)
        
        # Sélectionner seulement les colonnes nécessaires pour le ML
        ml_columns = [
            'city', 'price', 'fuel', 'boite_vitesse', 'first_hand',
            'brand_name', 'model_name', 'model_year', 'mileage'
        ]
        
        # Ajouter les colonnes qui existent dans le DataFrame
        available_columns = [col for col in ml_columns if col in df.columns]
        logger.info(f"Available columns for ML: {available_columns}")
        
        final_df = df[available_columns]
        logger.info(f"Final transformation result: {len(final_df)} records with {len(available_columns)} columns")
        
        return final_df
    
    def extract_new_cars(self, hours_back: int = 24) -> pd.DataFrame:
        """
        Extrait les nouvelles annonces ajoutées dans les dernières heures
        """
        try:
            # Calculer la date limite
            cutoff_date = datetime.now() - timedelta(hours=hours_back)
            
            # Query pour les nouvelles annonces
            query = {
                "datePosted": {"$gte": cutoff_date.isoformat()},
                "carPrice": {"$exists": True, "$ne": None}
            }
            
            # Projection des champs nécessaires
            projection = {
                "_id": 1,
                "carUri": 1,
                "label": 1,
                "carPrice": 1,
                "carCity": 1,
                "carFuel": 1,
                "carBV": 1,
                "carFirstHand": 1,
                "carBrand": 1,
                "carModel": 1,
                "carYear": 1,
                "carMileageMin": 1,
                "carMileageMax": 1,
                "datePosted": 1,
                "source": 1
            }
            
            cursor = self.collection.find(query, projection)
            data = list(cursor)
            
            if not data:
                logger.info("Aucune nouvelle annonce trouvée")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logger.info(f"Extraites {len(df)} nouvelles annonces")
            
            # Transformer les champs complexes
            df_transformed = self._transform_complex_fields(df)
            logger.info(f"Transformation terminée. Colonnes: {list(df_transformed.columns)}")
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {e}")
            return pd.DataFrame()
    
    def extract_all_cars(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Extrait toutes les annonces de voitures avec transformation
        """
        try:
            query = {"carPrice": {"$exists": True, "$ne": None}}
            
            projection = {
                "_id": 1,
                "carUri": 1,
                "label": 1,
                "carPrice": 1,
                "carCity": 1,
                "carFuel": 1,
                "carBV": 1,
                "carFirstHand": 1,
                "carBrand": 1,
                "carModel": 1,
                "carYear": 1,
                "carMileageMin": 1,
                "carMileageMax": 1,
                "datePosted": 1,
                "source": 1
            }
            
            cursor = self.collection.find(query, projection)
            if limit:
                cursor = cursor.limit(limit)
            
            data = list(cursor)
            
            if not data:
                logger.info("Aucune annonce trouvée")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logger.info(f"Extraites {len(df)} annonces")
            
            # Transformer les champs complexes
            df_transformed = self._transform_complex_fields(df)
            logger.info(f"Transformation terminée. Colonnes: {list(df_transformed.columns)}")
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {e}")
            return pd.DataFrame()
    
    def extract_raw_cars(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Extrait les données brutes sans transformation (pour debug)
        """
        try:
            query = {"carPrice": {"$exists": True, "$ne": None}}
            
            projection = {
                "_id": 1,
                "carUri": 1,
                "label": 1,
                "carPrice": 1,
                "carCity": 1,
                "carFuel": 1,
                "carBV": 1,
                "carFirstHand": 1,
                "carBrand": 1,
                "carModel": 1,
                "carYear": 1,
                "carMileageMin": 1,
                "carMileageMax": 1,
                "datePosted": 1,
                "source": 1
            }
            
            cursor = self.collection.find(query, projection)
            if limit:
                cursor = cursor.limit(limit)
            
            data = list(cursor)
            
            if not data:
                logger.info("Aucune annonce trouvée")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logger.info(f"Extraites {len(df)} annonces (raw)")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction: {e}")
            return pd.DataFrame()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur la collection
        """
        try:
            total_count = self.collection.count_documents({})
            with_price_count = self.collection.count_documents({"carPrice": {"$exists": True, "$ne": None}})
            
            # Dernière annonce ajoutée
            latest = self.collection.find_one(sort=[("datePosted", -1)])
            latest_date = latest.get("datePosted") if latest else None
            
            return {
                "total_documents": total_count,
                "documents_with_price": with_price_count,
                "latest_date": latest_date,
                "collection_name": Config.MONGODB_COLLECTION
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {}
    
    def close(self):
        """Ferme la connexion MongoDB"""
        self.client.close()

    def debug_field_extraction(self, limit: int = 3):
        """
        Debug function to analyze field extraction
        """
        try:
            cursor = self.collection.find({"carPrice": {"$exists": True, "$ne": None}}).limit(limit)
            data = list(cursor)
            
            if not data:
                print("No data found")
                return
            
            print("=== FIELD EXTRACTION DEBUG ===")
            for i, doc in enumerate(data):
                print(f"\n--- Document {i+1} ---")
                
                # Analyze each complex field
                complex_fields = ['carCity', 'carFuel', 'carBV', 'carFirstHand', 'carBrand', 'carModel']
                
                for field in complex_fields:
                    if field in doc:
                        raw_value = doc[field]
                        extracted_value = self._extract_value_from_dict(raw_value)
                        print(f"{field}:")
                        print(f"  Raw: {raw_value}")
                        print(f"  Extracted: {extracted_value}")
                        print()
                
                # Check price field
                if 'carPrice' in doc:
                    raw_price = doc['carPrice']
                    def extract_price_debug(x):
                        if x is None:
                            return None
                        if isinstance(x, dict) and 'value' in x and x.get('value') not in [None, '']:
                            return float(x.get('value'))
                        elif isinstance(x, (int, float)):
                            return float(x)
                        elif isinstance(x, str) and x.isdigit():
                            return float(x)
                        else:
                            return None
                    
                    price_value = extract_price_debug(raw_price)
                    print(f"carPrice:")
                    print(f"  Raw: {raw_price}")
                    print(f"  Extracted: {price_value}")
                    print()
                        
        except Exception as e:
            print(f"Error in debug: {e}")
            import traceback
            traceback.print_exc()

    def debug_data_pipeline(self, limit: int = 5):
        """
        Debug function to trace the entire data pipeline step by step
        """
        try:
            print("=== DATA PIPELINE DEBUG ===")
            
            # Step 1: Extract raw data
            print("\n1. Raw data extraction:")
            raw_df = self.extract_raw_cars(limit=limit)
            print(f"   Raw records: {len(raw_df)}")
            if not raw_df.empty:
                print(f"   Raw columns: {list(raw_df.columns)}")
            
            # Step 2: Transform complex fields
            print("\n2. Field transformation:")
            if not raw_df.empty:
                transformed_df = self._transform_complex_fields(raw_df)
                print(f"   Transformed records: {len(transformed_df)}")
                if not transformed_df.empty:
                    print(f"   Transformed columns: {list(transformed_df.columns)}")
                    
                    # Show sample data
                    print("\n3. Sample transformed data:")
                    print(transformed_df.head())
                    
                    # Show data types
                    print("\n4. Data types:")
                    print(transformed_df.dtypes)
                    
                    # Show null counts
                    print("\n5. Null value counts:")
                    print(transformed_df.isnull().sum())
                else:
                    print("   No records after transformation!")
            else:
                print("   No raw data to transform!")
                
        except Exception as e:
            print(f"Error in debug: {e}")
            import traceback
            traceback.print_exc()