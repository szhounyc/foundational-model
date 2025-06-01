#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Inference Service

Tests the enhanced inference service with Fireworks models, including:
- Model discovery and loading
- Chat completions
- Contract review functionality
- Batch processing
- Real data from processed datasets
"""

import json
import requests
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceServiceTester:
    """Test suite for the enhanced inference service"""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {details}")
    
    def test_service_health(self) -> bool:
        """Test if the service is running and healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                self.log_test_result(
                    "Service Health Check", 
                    True, 
                    f"Service running on {health_data.get('hardware', {}).get('device', 'unknown')}"
                )
                return True
            else:
                self.log_test_result("Service Health Check", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("Service Health Check", False, f"Error: {e}")
            return False
    
    def test_model_discovery(self) -> List[Dict]:
        """Test model discovery functionality"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=30)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("models", [])
                
                if models:
                    self.log_test_result(
                        "Model Discovery", 
                        True, 
                        f"Found {len(models)} models: {[m['id'] for m in models]}"
                    )
                    return models
                else:
                    self.log_test_result("Model Discovery", False, "No models found")
                    return []
            else:
                self.log_test_result("Model Discovery", False, f"Status: {response.status_code}")
                return []
        except Exception as e:
            self.log_test_result("Model Discovery", False, f"Error: {e}")
            return []
    
    def test_model_loading(self, model_id: str) -> bool:
        """Test loading a specific model"""
        try:
            # First check if already loaded
            response = requests.get(f"{self.base_url}/models/{model_id}/info", timeout=10)
            if response.status_code == 200:
                info = response.json()
                if info.get("status") == "loaded":
                    self.log_test_result("Model Loading", True, f"{model_id} already loaded")
                    return True
            
            # Load the model
            response = requests.post(f"{self.base_url}/models/{model_id}/load", timeout=300)
            if response.status_code == 200:
                result = response.json()
                self.log_test_result(
                    "Model Loading", 
                    True, 
                    f"Loaded {model_id}: {result.get('status')}"
                )
                return True
            else:
                self.log_test_result("Model Loading", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("Model Loading", False, f"Error: {e}")
            return False
    
    def test_basic_chat_completion(self, model_id: str) -> bool:
        """Test basic chat completion functionality"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you help me understand what you do?"}
            ]
            
            payload = {
                "model_id": model_id,
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("generated_text", "")
                
                self.log_test_result(
                    "Basic Chat Completion", 
                    True, 
                    f"Generated {len(generated_text)} chars in {result.get('generation_time', 0):.2f}s"
                )
                return True
            else:
                self.log_test_result("Basic Chat Completion", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("Basic Chat Completion", False, f"Error: {e}")
            return False
    
    def test_contract_review_functionality(self, model_id: str) -> bool:
        """Test specialized contract review endpoint"""
        try:
            contract_sections = [
                "Section 5.1: The closing date shall be no earlier than December 1, 2024.",
                "Section 7.1: Any notice under this Agreement shall be in writing and delivered by certified mail."
            ]
            
            payload = {
                "contract_sections": contract_sections,
                "model_id": model_id,
                "review_type": "comprehensive"
            }
            
            response = requests.post(f"{self.base_url}/contract/review", json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                review = result.get("review", "")
                
                self.log_test_result(
                    "Contract Review", 
                    True, 
                    f"Generated review: {len(review)} chars, {result.get('sections_reviewed')} sections"
                )
                return True
            else:
                self.log_test_result("Contract Review", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("Contract Review", False, f"Error: {e}")
            return False
    
    def load_test_data_from_datasets(self) -> List[Dict]:
        """Load test data from processed datasets"""
        test_cases = []
        
        # Load from MNTN contracts dataset
        dataset_path = Path("../feature-engineering/fireworks/processed_datasets/mntn_contracts_training.jsonl")
        
        if dataset_path.exists():
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 5:  # Limit to first 5 examples for testing
                            break
                        
                        data = json.loads(line.strip())
                        messages = data.get("messages", [])
                        
                        if len(messages) >= 2:
                            test_cases.append({
                                "name": f"MNTN Contract Example {i+1}",
                                "messages": messages[:2],  # System + User messages
                                "expected_type": "contract_review"
                            })
                
                logger.info(f"Loaded {len(test_cases)} test cases from dataset")
            except Exception as e:
                logger.error(f"Error loading test data: {e}")
        
        # Add some manual test cases if no dataset found
        if not test_cases:
            test_cases = [
                {
                    "name": "Simple Contract Question",
                    "messages": [
                        {"role": "system", "content": "You are a legal contract reviewer."},
                        {"role": "user", "content": "Please review this clause: 'The buyer shall close within 30 days.'"}
                    ],
                    "expected_type": "contract_review"
                },
                {
                    "name": "General Legal Question",
                    "messages": [
                        {"role": "system", "content": "You are a helpful legal assistant."},
                        {"role": "user", "content": "What should I consider when reviewing a real estate contract?"}
                    ],
                    "expected_type": "general"
                }
            ]
        
        return test_cases
    
    def test_with_real_data(self, model_id: str) -> bool:
        """Test the model with real data from processed datasets"""
        test_cases = self.load_test_data_from_datasets()
        
        if not test_cases:
            self.log_test_result("Real Data Testing", False, "No test data available")
            return False
        
        successful_tests = 0
        
        for test_case in test_cases:
            try:
                payload = {
                    "model_id": model_id,
                    "messages": test_case["messages"],
                    "max_tokens": 512,
                    "temperature": 0.3
                }
                
                response = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("generated_text", "")
                    
                    # Basic quality checks
                    is_quality_response = (
                        len(generated_text) > 20 and
                        not generated_text.lower().startswith("error") and
                        len(generated_text.split()) > 5
                    )
                    
                    if is_quality_response:
                        successful_tests += 1
                        logger.info(f"✅ {test_case['name']}: Generated {len(generated_text)} chars")
                    else:
                        logger.warning(f"⚠️ {test_case['name']}: Low quality response")
                
                else:
                    logger.error(f"❌ {test_case['name']}: HTTP {response.status_code}")
                
                # Small delay between requests
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ {test_case['name']}: Error - {e}")
        
        success_rate = successful_tests / len(test_cases)
        self.log_test_result(
            "Real Data Testing", 
            success_rate >= 0.7, 
            f"{successful_tests}/{len(test_cases)} tests passed ({success_rate:.1%})"
        )
        
        return success_rate >= 0.7
    
    def test_batch_processing(self, model_id: str) -> bool:
        """Test batch processing functionality"""
        try:
            # Create multiple requests
            requests_data = []
            for i in range(3):
                requests_data.append({
                    "model_id": model_id,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"This is test message {i+1}. Please respond briefly."}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.7
                })
            
            payload = {
                "requests": requests_data,
                "max_concurrent": 2
            }
            
            response = requests.post(f"{self.base_url}/batch/completions", json=payload, timeout=180)
            if response.status_code == 200:
                result = response.json()
                successful = result.get("successful", 0)
                total = result.get("total_requests", 0)
                
                self.log_test_result(
                    "Batch Processing", 
                    successful == total, 
                    f"{successful}/{total} batch requests successful"
                )
                return successful == total
            else:
                self.log_test_result("Batch Processing", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("Batch Processing", False, f"Error: {e}")
            return False
    
    def test_model_unloading(self, model_id: str) -> bool:
        """Test model unloading functionality"""
        try:
            response = requests.delete(f"{self.base_url}/models/{model_id}/unload", timeout=30)
            if response.status_code == 200:
                result = response.json()
                self.log_test_result(
                    "Model Unloading", 
                    True, 
                    f"Unloaded {model_id}: {result.get('status')}"
                )
                return True
            else:
                self.log_test_result("Model Unloading", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("Model Unloading", False, f"Error: {e}")
            return False
    
    def run_comprehensive_tests(self) -> Dict:
        """Run all tests and return summary"""
        print("🚀 Starting Enhanced Inference Service Tests")
        print("=" * 60)
        
        # Test 1: Service Health
        if not self.test_service_health():
            print("❌ Service not available. Please start the inference service first.")
            return self.get_test_summary()
        
        # Test 2: Model Discovery
        models = self.test_model_discovery()
        if not models:
            print("❌ No models found. Please ensure Fireworks models are available.")
            return self.get_test_summary()
        
        # Use the first available model for testing
        test_model = models[0]
        model_id = test_model["id"]
        
        print(f"\n🎯 Testing with model: {model_id}")
        print(f"   Type: {test_model.get('type')}")
        print(f"   Base Model: {test_model.get('base_model')}")
        
        # Test 3: Model Loading
        if not self.test_model_loading(model_id):
            print(f"❌ Failed to load model {model_id}")
            return self.get_test_summary()
        
        # Test 4: Basic Chat Completion
        self.test_basic_chat_completion(model_id)
        
        # Test 5: Contract Review
        self.test_contract_review_functionality(model_id)
        
        # Test 6: Real Data Testing
        self.test_with_real_data(model_id)
        
        # Test 7: Batch Processing
        self.test_batch_processing(model_id)
        
        # Test 8: Model Unloading (optional)
        # self.test_model_unloading(model_id)
        
        return self.get_test_summary()
    
    def get_test_summary(self) -> Dict:
        """Get summary of all test results"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["success"]])
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "results": self.test_results
        }
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        
        if summary['success_rate'] >= 0.8:
            print("🎉 Overall Status: EXCELLENT")
        elif summary['success_rate'] >= 0.6:
            print("✅ Overall Status: GOOD")
        else:
            print("⚠️ Overall Status: NEEDS IMPROVEMENT")
        
        return summary

def main():
    """Main test execution"""
    tester = InferenceServiceTester()
    
    try:
        summary = tester.run_comprehensive_tests()
        
        # Save results
        with open("inference_test_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Test results saved to: inference_test_results.json")
        
        # Exit with appropriate code
        exit_code = 0 if summary["success_rate"] >= 0.7 else 1
        exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        exit(1)

if __name__ == "__main__":
    main() 