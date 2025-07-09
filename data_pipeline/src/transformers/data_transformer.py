import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self):
        # Mapping des champs Java vers Python
        self.field_mapping = {
            'carCity': 'city',
            'carPrice': 'price', 
            'carFuel': 'fuel',
            'carBV': 'boite_vitesse',
            'carFirstHand': 'first_hand',
            'carBrand': 'brand_name',
            'carModel': 'model_name',
            'carYear': 'model_year',
            'carMileageMin': 'mileage_min',
            'carMileageMax': 'mileage_max'
        }
        
        # Mapping des valeurs pour first_hand
        self.first_hand_mapping = {
            'oui': 1,
            'non': 0,
            'yes': 1,
            'no': 0,
            True: 1,
            False: 0,
            '1': 1,
            '0': 0,
            1: 1,
            0: 0
        }
        
        # Top cities based on notebook analysis
        self.top_cities = [
            'casablanca', 'rabat', 'tanger', 'marrakech', 'fès', 'agadir', 
            'kénitra', 'meknès', 'salé', 'tétouan', 'temara', 'el jadida',
            'mohammedia', 'oujda', 'nador', 'safi', 'khouribga', 'béni mellal',
            'berrechid', 'laâyoune', 'settat', 'bouskoura', 'taza', 'larache',
            'dakhla', 'ouarzazate', 'khemisset', 'essaouira', 'fquih ben saleh',
            'khénifra', 'errachidia', 'bouznika', 'berkane', 'ben guerir',
            'tifelt', 'el kelâa des sraghna', 'guelmim', 'benslimane',
            'sidi slimane', 'skhirat', 'ksar el kebir', 'sidi kacem',
            'sidi bennour', 'guercif', 'sefrou', 'taroudant', 'tiznit',
            'ouazzane', 'el hajeb', 'tan tan', 'had soualem', 'deroua',
            'azrou', 'dar bouazza', 'nouaceur', 'oued zem', 'ain aouda',
            'ifrane', 'midelt', 'al hoceima'
        ]
    
    def _extract_value_from_dict(self, value: Any) -> Optional[str]:
        """
        Extrait la valeur d'un dictionnaire MongoDB ou retourne la valeur directement
        """
        if value is None or pd.isna(value):
            return None
        
        # Si c'est un dictionnaire, extraire la valeur
        if isinstance(value, dict):
            # Priorité: 'value' > 'label' > 'key'
            if 'value' in value and value['value'] is not None:
                return str(value['value']).strip()
            elif 'label' in value and value['label'] is not None:
                return str(value['label']).strip()
            elif 'key' in value and value['key'] is not None:
                return str(value['key']).strip()
            else:
                return None
        
        # Si c'est déjà une chaîne ou un autre type simple
        return str(value).strip()
    
    def transform_mongodb_to_ml_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme les données MongoDB au format attendu par le modèle ML
        """
        if df.empty:
            logger.warning("DataFrame vide fourni à la transformation")
            return pd.DataFrame()
        
        try:
            logger.info(f"Début de la transformation avec {len(df)} enregistrements")
            logger.debug(f"Colonnes d'entrée: {list(df.columns)}")
            
            # Créer une copie pour éviter les modifications inattendues
            df_transformed = df.copy()
            
            # 1. Supprimer les colonnes inutiles (comme dans le notebook)
            columns_to_drop = ['id', 'date_annonce', 'source_uri', 'image', 'is_last', 'car_name']
            df_transformed = df_transformed.drop(columns=[col for col in columns_to_drop if col in df_transformed.columns])
            
            # 2. Mapper les noms de colonnes si nécessaire
            df_transformed = df_transformed.rename(columns=self.field_mapping)
            
            # 3. Extraire les valeurs des dictionnaires MongoDB
            df_transformed = self._extract_mongodb_values(df_transformed)
            
            # 4. Supprimer les lignes avec des valeurs manquantes critiques (comme dans le notebook)
            critical_columns = ['city', 'fuel', 'boite_vitesse', 'brand_name', 'model_name', 'model_year']
            available_critical = [col for col in critical_columns if col in df_transformed.columns]
            if available_critical:
                initial_count = len(df_transformed)
                df_transformed = df_transformed.dropna(subset=available_critical)
                logger.info(f"Suppression valeurs manquantes critiques: {initial_count} -> {len(df_transformed)}")
            
            # 5. Créer la colonne mileage à partir de min et max
            df_transformed = self._create_mileage_column(df_transformed)
            
            # 6. Nettoyer les données (normalisation texte, types, etc.)
            df_transformed = self._clean_data(df_transformed)
            
            # 7. Appliquer les filtres spécifiques (prix, kilométrage, années)
            df_transformed = self._apply_specific_cleaning(df_transformed)
            
            # 8. Filtrer les villes (comme dans le notebook)
            df_transformed = self._filter_cities(df_transformed)
            
            # 9. Filtrer les modèles (optionnel - basé sur le notebook)
            df_transformed = self._filter_models(df_transformed)
            
            # 10. Sélectionner uniquement les colonnes nécessaires pour le modèle
            required_columns = ['city', 'fuel', 'boite_vitesse', 'brand_name', 
                              'model_name', 'model_year', 'mileage', 'first_hand']
            
            # Ajouter la colonne price si elle existe
            if 'price' in df_transformed.columns:
                required_columns.insert(1, 'price')
            
            # Vérifier que toutes les colonnes requises sont présentes
            missing_columns = set(required_columns) - set(df_transformed.columns)
            if missing_columns:
                logger.warning(f"Colonnes manquantes: {missing_columns}")
                for col in missing_columns:
                    df_transformed[col] = None
            
            # Sélectionner les colonnes dans l'ordre correct
            available_columns = [col for col in required_columns if col in df_transformed.columns]
            df_final = df_transformed[available_columns].copy()
            
            logger.info(f"Transformation terminée. {len(df_final)} lignes restantes")
            return df_final
            
        except Exception as e:
            logger.error(f"Erreur lors de la transformation: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _extract_mongodb_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrait les valeurs des dictionnaires MongoDB
        """
        # Colonnes qui contiennent des dictionnaires
        dict_columns = ['city', 'fuel', 'boite_vitesse', 'brand_name', 'model_name', 'first_hand']
        
        for col in dict_columns:
            if col in df.columns:
                logger.debug(f"Extraction des valeurs pour {col}")
                df[col] = df[col].apply(self._extract_value_from_dict)
        
        return df
    
    def _create_mileage_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée la colonne mileage à partir de mileage_min et mileage_max
        Utilise la même logique que dans le notebook
        """
        if 'mileage_min' in df.columns and 'mileage_max' in df.columns:
            def clean_mileage(row):
                min_val = row.get('mileage_min')
                max_val = row.get('mileage_max')
                
                # Convertir en float si possible
                try:
                    min_val = float(min_val) if min_val is not None else None
                    max_val = float(max_val) if max_val is not None else None
                except (ValueError, TypeError):
                    return None
                
                # Si les deux valeurs sont None
                if min_val is None and max_val is None:
                    return None
                
                # Si une seule valeur est disponible
                if min_val is None:
                    return max_val
                if max_val is None:
                    return min_val
                
                # Si min > max, il y a une erreur - prendre le minimum
                if min_val > max_val:
                    return min_val
                
                # Si min = max, kilométrage exact
                if min_val == max_val:
                    return min_val
                
                # Sinon, prendre la moyenne de la fourchette
                return (min_val + max_val) / 2
            
            df['mileage'] = df.apply(clean_mileage, axis=1)
            
            # Supprimer les anciennes colonnes (comme dans le notebook)
            df = df.drop(['mileage_min', 'mileage_max'], axis=1)
            logger.debug(f"Colonne mileage créée: {df['mileage'].notna().sum()} valeurs valides")
        elif 'mileage' not in df.columns:
            # Si ni mileage_min ni mileage_max n'existent, créer une colonne vide
            df['mileage'] = None
            logger.warning("Colonnes mileage_min/max manquantes, mileage défini à None")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le nettoyage général aux données (même logique que le notebook)
        """
        # Nettoyer les champs texte
        text_columns = ['city', 'fuel', 'boite_vitesse', 'brand_name', 'model_name']
        for col in text_columns:
            if col in df.columns:
                # Convertir en string et nettoyer
                df[col] = df[col].astype(str)
                df[col] = df[col].str.strip().str.lower()
                df[col] = df[col].str.replace('-', ' ', regex=False)
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
                # Remplacer 'none' et 'nan' par None
                df[col] = df[col].replace(['none', 'nan', 'null'], None)
        
        # Nettoyer first_hand
        if 'first_hand' in df.columns:
            # Convertir d'abord en string pour uniformiser
            df['first_hand'] = df['first_hand'].astype(str).str.lower()
            df['first_hand'] = df['first_hand'].map(self.first_hand_mapping)
            # Remplir les valeurs manquantes avec 0
            df['first_hand'] = df['first_hand'].fillna(0)
        
        # Nettoyer les valeurs numériques
        numeric_columns = ['price', 'model_year', 'mileage']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _apply_specific_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le nettoyage spécifique (filtres sur prix, kilométrage, années)
        """
        if df.empty:
            logger.warning("DataFrame vide pour le nettoyage spécifique")
            return df
        
        # Filtrer les prix raisonnables
        if 'price' in df.columns and not df.empty:
            initial_count = len(df)
            df = df[(df['price'] >= 10000) & (df['price'] <= 900000)]
            logger.info(f"Filtrage des prix: {initial_count} -> {len(df)} lignes")
        
        # Filtrer les kilométrages raisonnables
        if 'mileage' in df.columns and not df.empty:
            initial_count = len(df)
            df = df[(df['mileage'] >= 1) & (df['mileage'] <= 600000)]
            logger.info(f"Filtrage du kilométrage: {initial_count} -> {len(df)} lignes")
        
        # Filtrer les années raisonnables
        if 'model_year' in df.columns and not df.empty:
            initial_count = len(df)
            current_year = 2025
            df = df[(df['model_year'] >= 1990) & (df['model_year'] <= current_year)]
            logger.info(f"Filtrage des années: {initial_count} -> {len(df)} lignes")
        
        # Normaliser les valeurs fuel (comme dans le notebook)
        if 'fuel' in df.columns:
            fuel_mapping = {
                'gasoline': 'essence',
                'electric': 'electrique', 
                'hybrid': 'hybride',
                'dieselmhev': 'diesel',
                'essencemhev': 'essence',
                'hybriderechargeable': 'hybride'
            }
            df['fuel'] = df['fuel'].replace(fuel_mapping)
        
        # Normaliser les valeurs boite_vitesse (comme dans le notebook)
        if 'boite_vitesse' in df.columns:
            df['boite_vitesse'] = df['boite_vitesse'].replace({'man': 'manuelle'})
        
        # Convertir les types finaux
        if 'model_year' in df.columns and not df.empty:
            df['model_year'] = df['model_year'].astype(int)
        if 'price' in df.columns and not df.empty:
            df['price'] = df['price'].astype(int)
        if 'first_hand' in df.columns:
            df['first_hand'] = df['first_hand'].fillna(0).astype(int)
        
        return df
    
    def _filter_cities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtrage des villes temporairement désactivé pour debug
        """
        if 'city' in df.columns and not df.empty:
            initial_count = len(df)
            logger.info(f"Filtrage des villes désactivé: {initial_count} -> {initial_count} lignes")
        return df
    
    def _filter_models(self, df: pd.DataFrame, top_n=700) -> pd.DataFrame:
        """
        Filtre les modèles comme dans le notebook (garde seulement les top models)
        """
        if 'model_name' in df.columns and not df.empty:
            initial_count = len(df)
            top_models = df['model_name'].value_counts().head(top_n).index
            df = df[df['model_name'].isin(top_models)]
            logger.info(f"Filtrage des modèles (top {top_n}): {initial_count} -> {len(df)} lignes")
        return df