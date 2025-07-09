import logging
from pipelines.etl_pipeline import ETLPipeline

if __name__ == "__main__":
    # Configuration des logs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Créer le pipeline
    pipeline = ETLPipeline()
    
    # Tester les statistiques de la DB
    print("=== Statistiques de la base de données ===")
    db_stats = pipeline.get_db_stats()
    print(db_stats)
    
    # Tester le pipeline incrémental
    print("\n=== Test du pipeline incrémental ===")
    result = pipeline.run_incremental_pipeline(hours_back=168)  # 7 jours
    print(result)
    
    # Fermer les connexions
    pipeline.close()