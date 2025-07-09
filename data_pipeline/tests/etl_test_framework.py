import logging
import sys
import traceback
from typing import Dict, Any, Optional
import pandas as pd
from pipelines.etl_pipeline import ETLPipeline
from extractors.mongo_extractor import MongoExtractor
from transformers.data_transformer import DataTransformer

class ETLTester:
    def __init__(self):
        self.setup_logging()
        self.pipeline = None
        self.test_results = {}
    
    def setup_logging(self):
        """Configure logging for testing"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('etl_test.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def test_database_connection(self) -> Dict[str, Any]:
        """Test 1: Database connection and basic stats"""
        self.logger.info("🔍 Testing database connection...")
        
        try:
            extractor = MongoExtractor()
            stats = extractor.get_collection_stats()
            extractor.close()
            
            if stats and stats.get('total_documents', 0) > 0:
                result = {
                    "status": "✅ PASS",
                    "message": "Database connection successful",
                    "stats": stats
                }
            else:
                result = {
                    "status": "❌ FAIL",
                    "message": "Database connection failed or no data found",
                    "stats": stats
                }
            
            self.test_results["database_connection"] = result
            return result
            
        except Exception as e:
            result = {
                "status": "❌ FAIL",
                "message": f"Database connection error: {str(e)}",
                "error": traceback.format_exc()
            }
            self.test_results["database_connection"] = result
            return result
    
    def test_data_extraction(self, limit: int = 10) -> Dict[str, Any]:
        """Test 2: Data extraction functionality"""
        self.logger.info("🔍 Testing data extraction...")
        
        try:
            extractor = MongoExtractor()
            
            # Test raw extraction
            raw_data = extractor.extract_raw_cars(limit=limit)
            
            # Test transformed extraction
            transformed_data = extractor.extract_all_cars(limit=limit)
            
            # Test new cars extraction
            new_data = extractor.extract_new_cars(hours_back=24)
            
            extractor.close()
            
            result = {
                "status": "✅ PASS",
                "message": "Data extraction successful",
                "raw_data_count": len(raw_data),
                "transformed_data_count": len(transformed_data),
                "new_data_count": len(new_data),
                "sample_columns": list(raw_data.columns) if not raw_data.empty else [],
                "transformed_columns": list(transformed_data.columns) if not transformed_data.empty else []
            }
            
            if raw_data.empty and transformed_data.empty:
                result["status"] = "⚠️ WARNING"
                result["message"] = "No data extracted - check your MongoDB query"
            
            self.test_results["data_extraction"] = result
            return result
            
        except Exception as e:
            result = {
                "status": "❌ FAIL",
                "message": f"Data extraction error: {str(e)}",
                "error": traceback.format_exc()
            }
            self.test_results["data_extraction"] = result
            return result
    
    def test_data_transformation(self, limit: int = 10) -> Dict[str, Any]:
        """Test 3: Data transformation functionality"""
        self.logger.info("🔍 Testing data transformation...")
        
        try:
            extractor = MongoExtractor()
            transformer = DataTransformer()
            
            # Get raw data
            raw_data = extractor.extract_raw_cars(limit=limit)
            
            if raw_data.empty:
                result = {
                    "status": "⚠️ SKIP",
                    "message": "No raw data available for transformation testing"
                }
                self.test_results["data_transformation"] = result
                return result
            
            # Transform data
            transformed_data = transformer.transform_mongodb_to_ml_format(raw_data)
            
            # Check transformation results
            expected_columns = ['city', 'fuel', 'boite_vitesse', 'brand_name', 
                              'model_name', 'model_year', 'mileage', 'first_hand']
            
            missing_columns = set(expected_columns) - set(transformed_data.columns)
            extra_columns = set(transformed_data.columns) - set(expected_columns)
            
            result = {
                "status": "✅ PASS",
                "message": "Data transformation successful",
                "input_shape": raw_data.shape,
                "output_shape": transformed_data.shape,
                "expected_columns": expected_columns,
                "actual_columns": list(transformed_data.columns),
                "missing_columns": list(missing_columns),
                "extra_columns": list(extra_columns)
            }
            
            if missing_columns:
                result["status"] = "⚠️ WARNING"
                result["message"] = f"Missing expected columns: {missing_columns}"
            
            if transformed_data.empty:
                result["status"] = "❌ FAIL"
                result["message"] = "Transformation resulted in empty dataset"
            
            extractor.close()
            self.test_results["data_transformation"] = result
            return result
            
        except Exception as e:
            result = {
                "status": "❌ FAIL",
                "message": f"Data transformation error: {str(e)}",
                "error": traceback.format_exc()
            }
            self.test_results["data_transformation"] = result
            return result
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test 4: Model loading"""
        self.logger.info("🔍 Testing model loading...")
        
        try:
            pipeline = ETLPipeline()
            
            if pipeline.model is not None:
                result = {
                    "status": "✅ PASS",
                    "message": "Model loaded successfully",
                    "model_type": str(type(pipeline.model))
                }
            else:
                result = {
                    "status": "❌ FAIL",
                    "message": "Model loading failed - model is None"
                }
            
            pipeline.close()
            self.test_results["model_loading"] = result
            return result
            
        except Exception as e:
            result = {
                "status": "❌ FAIL",
                "message": f"Model loading error: {str(e)}",
                "error": traceback.format_exc()
            }
            self.test_results["model_loading"] = result
            return result
    
    def test_full_pipeline(self, limit: int = 5) -> Dict[str, Any]:
        """Test 5: Full pipeline execution"""
        self.logger.info("🔍 Testing full pipeline...")
        
        try:
            pipeline = ETLPipeline()
            result_data = pipeline.run_full_pipeline(limit=limit)
            
            result = {
                "status": "✅ PASS" if result_data.get("status") == "success" else "❌ FAIL",
                "message": "Full pipeline execution completed",
                "pipeline_result": result_data
            }
            
            pipeline.close()
            self.test_results["full_pipeline"] = result
            return result
            
        except Exception as e:
            result = {
                "status": "❌ FAIL",
                "message": f"Full pipeline error: {str(e)}",
                "error": traceback.format_exc()
            }
            self.test_results["full_pipeline"] = result
            return result
    
    def test_incremental_pipeline(self, hours_back: int = 168) -> Dict[str, Any]:
        """Test 6: Incremental pipeline execution"""
        self.logger.info("🔍 Testing incremental pipeline...")
        
        try:
            pipeline = ETLPipeline()
            result_data = pipeline.run_incremental_pipeline(hours_back=hours_back)
            
            result = {
                "status": "✅ PASS" if result_data.get("status") in ["success", "info"] else "❌ FAIL",
                "message": "Incremental pipeline execution completed",
                "pipeline_result": result_data
            }
            
            pipeline.close()
            self.test_results["incremental_pipeline"] = result
            return result
            
        except Exception as e:
            result = {
                "status": "❌ FAIL",
                "message": f"Incremental pipeline error: {str(e)}",
                "error": traceback.format_exc()
            }
            self.test_results["incremental_pipeline"] = result
            return result
    
    def test_field_extraction_debug(self) -> Dict[str, Any]:
        """Test 7: Field extraction debugging"""
        self.logger.info("🔍 Testing field extraction debugging...")
        
        try:
            extractor = MongoExtractor()
            
            # Capture debug output
            import io
            from contextlib import redirect_stdout
            
            debug_output = io.StringIO()
            with redirect_stdout(debug_output):
                extractor.debug_field_extraction(limit=2)
            
            debug_text = debug_output.getvalue()
            
            result = {
                "status": "✅ PASS",
                "message": "Field extraction debug completed",
                "debug_output": debug_text[:1000] + "..." if len(debug_text) > 1000 else debug_text
            }
            
            extractor.close()
            self.test_results["field_extraction_debug"] = result
            return result
            
        except Exception as e:
            result = {
                "status": "❌ FAIL",
                "message": f"Field extraction debug error: {str(e)}",
                "error": traceback.format_exc()
            }
            self.test_results["field_extraction_debug"] = result
            return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🚀 Starting ETL Pipeline Testing Suite")
        print("=" * 60)
        
        tests = [
            ("Database Connection", self.test_database_connection),
            ("Data Extraction", lambda: self.test_data_extraction(limit=10)),
            ("Data Transformation", lambda: self.test_data_transformation(limit=10)),
            ("Model Loading", self.test_model_loading),
            ("Full Pipeline", lambda: self.test_full_pipeline(limit=5)),
            ("Incremental Pipeline", lambda: self.test_incremental_pipeline(hours_back=168)),
            ("Field Extraction Debug", self.test_field_extraction_debug)
        ]
        
        for test_name, test_func in tests:
            print(f"\n📋 Running: {test_name}")
            try:
                result = test_func()
                print(f"   {result['status']}: {result['message']}")
                if result['status'] == "❌ FAIL" and 'error' in result:
                    print(f"   Error details: {result['error'][:200]}...")
            except Exception as e:
                print(f"   ❌ FAIL: Unexpected error in {test_name}: {str(e)}")
        
        # Generate summary
        summary = self.generate_summary()
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(summary)
        
        return self.test_results
    
    def generate_summary(self) -> str:
        """Generate a summary of all test results"""
        if not self.test_results:
            return "No tests executed"
        
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r['status'] == '✅ PASS')
        failed = sum(1 for r in self.test_results.values() if r['status'] == '❌ FAIL')
        warnings = sum(1 for r in self.test_results.values() if r['status'] == '⚠️ WARNING')
        skipped = sum(1 for r in self.test_results.values() if r['status'] == '⚠️ SKIP')
        
        summary = f"""
Total Tests: {total_tests}
✅ Passed: {passed}
❌ Failed: {failed}
⚠️ Warnings: {warnings}
⚠️ Skipped: {skipped}

Success Rate: {(passed/total_tests)*100:.1f}%
"""
        
        if failed > 0:
            summary += "\n🔥 FAILED TESTS:\n"
            for test_name, result in self.test_results.items():
                if result['status'] == '❌ FAIL':
                    summary += f"- {test_name}: {result['message']}\n"
        
        if warnings > 0:
            summary += "\n⚠️ WARNINGS:\n"
            for test_name, result in self.test_results.items():
                if result['status'] == '⚠️ WARNING':
                    summary += f"- {test_name}: {result['message']}\n"
        
        return summary

def main():
    """Main testing function"""
    tester = ETLTester()
    results = tester.run_all_tests()
    
    # Save results to file
    import json
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to 'test_results.json'")
    print(f"📋 Test log saved to 'etl_test.log'")

if __name__ == "__main__":
    main()