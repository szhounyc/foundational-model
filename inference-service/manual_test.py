#!/usr/bin/env python3
"""
Manual Test Script for Enhanced Inference Service
Tests all endpoints and provides detailed feedback
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8003"

def test_endpoint(name: str, method: str, endpoint: str, data: Dict[Any, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Test a single endpoint and return results"""
    url = f"{BASE_URL}{endpoint}"
    
    if headers is None:
        headers = {"Content-Type": "application/json"}
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        elif method.upper() == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}
        
        result = {
            "status": "success" if response.status_code < 400 else "error",
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds(),
        }
        
        try:
            result["data"] = response.json()
        except:
            result["data"] = response.text
            
        return result
        
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

def print_test_result(name: str, result: Dict[str, Any]):
    """Print formatted test result"""
    status_emoji = "✅" if result["status"] == "success" else "❌"
    print(f"{status_emoji} {name}")
    print(f"   Status: {result.get('status_code', 'N/A')} ({result['status']})")
    print(f"   Time: {result.get('response_time', 0):.3f}s")
    
    if result["status"] == "error" and "message" in result:
        print(f"   Error: {result['message']}")
    elif "data" in result:
        if isinstance(result["data"], dict):
            print(f"   Response: {json.dumps(result['data'], indent=6)}")
        else:
            print(f"   Response: {result['data'][:200]}...")
    print()

def main():
    """Run all tests"""
    print("🚀 Enhanced Inference Service Manual Tests")
    print("=" * 50)
    print()
    
    # Test 1: Health Check
    result = test_endpoint("Health Check", "GET", "/health")
    print_test_result("Health Check", result)
    
    # Test 2: Root Endpoint
    result = test_endpoint("Root Endpoint", "GET", "/")
    print_test_result("Root Endpoint", result)
    
    # Test 3: Models List
    result = test_endpoint("Models List", "GET", "/models")
    print_test_result("Models List", result)
    
    # Test 4: Try to load a model (this might fail but we'll see the error)
    result = test_endpoint("Load Model (sftj-s1xkr35z)", "POST", "/models/sftj-s1xkr35z/load")
    print_test_result("Load Model (sftj-s1xkr35z)", result)
    
    # Test 5: Try alternative model ID
    result = test_endpoint("Load Model (zlg-re-fm-sl-mntn)", "POST", "/models/zlg-re-fm-sl-mntn/load")
    print_test_result("Load Model (zlg-re-fm-sl-mntn)", result)
    
    # Test 6: Chat Completion (will likely fail without loaded model)
    chat_data = {
        "model_id": "base",
        "messages": [
            {"role": "user", "content": "Hello, this is a test message. Please respond briefly."}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    result = test_endpoint("Chat Completion", "POST", "/chat/completions", chat_data)
    print_test_result("Chat Completion", result)
    
    # Test 7: Contract Review
    contract_data = {
        "contract_sections": [
            "Section 1: The Buyer agrees to purchase the property for $500,000.",
            "Section 2: Closing shall occur within 30 days of contract execution."
        ],
        "review_type": "quick"
    }
    result = test_endpoint("Contract Review", "POST", "/contract/review", contract_data)
    print_test_result("Contract Review", result)
    
    # Test 8: Batch Completions
    batch_data = {
        "requests": [
            {
                "model_id": "base",
                "messages": [{"role": "user", "content": "Test 1"}],
                "max_tokens": 20
            },
            {
                "model_id": "base", 
                "messages": [{"role": "user", "content": "Test 2"}],
                "max_tokens": 20
            }
        ]
    }
    result = test_endpoint("Batch Completions", "POST", "/batch/completions", batch_data)
    print_test_result("Batch Completions", result)
    
    # Test 9: Model Info (for a model that might exist)
    result = test_endpoint("Model Info", "GET", "/models/base/info")
    print_test_result("Model Info", result)
    
    print("🏁 Test Summary")
    print("=" * 50)
    print("All endpoint tests completed!")
    print("Check the results above to see which endpoints are working.")
    print()
    print("💡 Tips:")
    print("- If model loading fails, check the models directory structure")
    print("- If chat/contract endpoints fail, ensure a model is loaded first")
    print("- Check service logs for detailed error information")

if __name__ == "__main__":
    main() 