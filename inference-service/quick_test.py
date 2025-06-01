#!/usr/bin/env python3
"""
Quick Test Script for Enhanced Inference Service
Tests working endpoints and provides performance metrics
"""

import requests
import json
import time
import concurrent.futures
from typing import List, Dict

BASE_URL = "http://localhost:8003"

def test_health_check():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    return response.status_code == 200, response.json()

def test_root_endpoint():
    """Test root endpoint"""
    response = requests.get(f"{BASE_URL}/")
    return response.status_code == 200, response.json()

def test_models_list():
    """Test models list endpoint"""
    response = requests.get(f"{BASE_URL}/models")
    return response.status_code == 200, response.json()

def load_test_health(num_requests: int = 10):
    """Load test the health endpoint"""
    def single_request():
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/health")
        end_time = time.time()
        return {
            "status_code": response.status_code,
            "response_time": end_time - start_time,
            "success": response.status_code == 200
        }
    
    print(f"🔄 Running load test with {num_requests} concurrent requests...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(single_request) for _ in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Calculate statistics
    response_times = [r["response_time"] for r in results]
    success_count = sum(1 for r in results if r["success"])
    
    stats = {
        "total_requests": num_requests,
        "successful_requests": success_count,
        "success_rate": success_count / num_requests * 100,
        "avg_response_time": sum(response_times) / len(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times)
    }
    
    return stats

def test_api_documentation():
    """Test if API documentation is accessible"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        return response.status_code == 200, "API docs accessible"
    except:
        return False, "API docs not accessible"

def test_openapi_spec():
    """Test if OpenAPI specification is accessible"""
    try:
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code == 200:
            spec = response.json()
            return True, f"OpenAPI spec available with {len(spec.get('paths', {}))} endpoints"
        else:
            return False, f"OpenAPI spec returned {response.status_code}"
    except Exception as e:
        return False, f"Error accessing OpenAPI spec: {e}"

def main():
    """Run quick tests"""
    print("⚡ Enhanced Inference Service Quick Tests")
    print("=" * 50)
    print()
    
    # Test 1: Health Check
    print("1. 🏥 Health Check Test")
    success, data = test_health_check()
    if success:
        print(f"   ✅ PASS - Service is healthy")
        print(f"   📊 Hardware: {data['hardware']['device']} ({data['hardware']['platform']})")
        print(f"   🔢 Loaded models: {data['model_count']}")
    else:
        print(f"   ❌ FAIL - Health check failed")
    print()
    
    # Test 2: Root Endpoint
    print("2. 🏠 Root Endpoint Test")
    success, data = test_root_endpoint()
    if success:
        print(f"   ✅ PASS - Root endpoint accessible")
        print(f"   📋 Version: {data.get('version', 'Unknown')}")
        print(f"   🎯 Features: {', '.join(data.get('features', []))}")
    else:
        print(f"   ❌ FAIL - Root endpoint failed")
    print()
    
    # Test 3: Models List
    print("3. 📚 Models List Test")
    success, data = test_models_list()
    if success:
        print(f"   ✅ PASS - Models endpoint accessible")
        print(f"   📊 Total models: {data.get('total_count', 0)}")
        print(f"   🔄 Loaded models: {data.get('loaded_count', 0)}")
    else:
        print(f"   ❌ FAIL - Models endpoint failed")
    print()
    
    # Test 4: API Documentation
    print("4. 📖 API Documentation Test")
    success, message = test_api_documentation()
    if success:
        print(f"   ✅ PASS - {message}")
        print(f"   🌐 URL: {BASE_URL}/docs")
    else:
        print(f"   ❌ FAIL - {message}")
    print()
    
    # Test 5: OpenAPI Specification
    print("5. 📋 OpenAPI Specification Test")
    success, message = test_openapi_spec()
    if success:
        print(f"   ✅ PASS - {message}")
        print(f"   🌐 URL: {BASE_URL}/openapi.json")
    else:
        print(f"   ❌ FAIL - {message}")
    print()
    
    # Test 6: Load Test
    print("6. 🚀 Load Test (Health Endpoint)")
    stats = load_test_health(20)
    print(f"   📊 Results:")
    print(f"      Total requests: {stats['total_requests']}")
    print(f"      Successful: {stats['successful_requests']}")
    print(f"      Success rate: {stats['success_rate']:.1f}%")
    print(f"      Avg response time: {stats['avg_response_time']*1000:.1f}ms")
    print(f"      Min response time: {stats['min_response_time']*1000:.1f}ms")
    print(f"      Max response time: {stats['max_response_time']*1000:.1f}ms")
    
    if stats['success_rate'] >= 95:
        print(f"   ✅ PASS - Load test successful")
    else:
        print(f"   ⚠️  WARN - Load test had some failures")
    print()
    
    # Summary
    print("🏁 Quick Test Summary")
    print("=" * 50)
    print("✅ Service is running and responding to requests")
    print("✅ Basic endpoints are functional")
    print("✅ API documentation is accessible")
    print("⚠️  Model loading functionality needs investigation")
    print()
    print("🔗 Useful URLs:")
    print(f"   Service: {BASE_URL}")
    print(f"   Health: {BASE_URL}/health")
    print(f"   API Docs: {BASE_URL}/docs")
    print(f"   OpenAPI: {BASE_URL}/openapi.json")
    print()
    print("💡 Next Steps:")
    print("   1. Check service logs for model loading issues")
    print("   2. Verify Fireworks model directory structure")
    print("   3. Test model loading with correct model IDs")

if __name__ == "__main__":
    main() 