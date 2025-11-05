"""
Test script for SVM Character Classifier
Tests all SVM endpoints to verify functionality
"""

import requests
import json
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

BASE_URL = "http://localhost:5000"

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")
    print("="*60)

def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def print_info(text):
    """Print info message"""
    print(f"{Fore.YELLOW}ℹ {text}{Style.RESET_ALL}")

def test_health():
    """Test 1: Check ML service health including SVM"""
    print_header("Test 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        if response.status_code == 200:
            print_success(f"Service Status: {data['status']}")
            print_success(f"Service Name: {data['service']}")
            
            models = data['models']
            print_info(f"K-NN: {'✓ Loaded' if models['knn']['loaded'] else '✗ Not loaded'}, {'✓ Trained' if models['knn']['trained'] else '✗ Not trained'}")
            print_info(f"Linear Regression: {'✓ Loaded' if models['lr']['loaded'] else '✗ Not loaded'}, {'✓ Trained' if models['lr']['trained'] else '✗ Not trained'}")
            print_info(f"Naive Bayes: {'✓ Loaded' if models['nb']['loaded'] else '✗ Not loaded'}, {'✓ Trained' if models['nb']['trained'] else '✗ Not trained'}")
            print_info(f"SVM: {'✓ Loaded' if models['svm']['loaded'] else '✗ Not loaded'}, {'✓ Trained' if models['svm']['trained'] else '✗ Not trained'}")
            
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_svm_training():
    """Test 2: Train SVM model"""
    print_header("Test 2: Train SVM Model")
    try:
        print_info("Training SVM with RBF kernel (this may take 30-60 seconds)...")
        response = requests.post(
            f"{BASE_URL}/train-svm",
            json={
                "kernel": "rbf",
                "optimize": False  # Set to True for GridSearchCV optimization (slower)
            }
        )
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            print_success("SVM model trained successfully!")
            
            metrics = data['metrics']
            print_info(f"Training Accuracy: {metrics['train_accuracy']:.2%}")
            print_info(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
            print_info(f"Kernel: {metrics['kernel']}")
            print_info(f"Number of Classes: {metrics['n_classes']}")
            print_info(f"Support Vectors: {metrics['n_support_vectors']}")
            print_info(f"Training Samples: {metrics['n_training_samples']}")
            print_info(f"Test Samples: {metrics['n_test_samples']}")
            
            return True
        else:
            print_error(f"Training failed: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print_error(f"Training failed: {e}")
        return False

def test_svm_prediction():
    """Test 3: Predict character using SVM"""
    print_header("Test 3: SVM Character Prediction")
    
    # Test cases
    test_cases = [
        {
            "text": "With great power comes great responsibility",
            "description": "Spider-Man's iconic quote"
        },
        {
            "text": "I am vengeance, I am the night, I am Batman!",
            "description": "Batman's declaration"
        },
        {
            "text": "I can do this all day",
            "description": "Captain America's determination"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{Fore.YELLOW}Test Case {i}: {test_case['description']}{Style.RESET_ALL}")
        print(f"Input: \"{test_case['text']}\"")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict-svm",
                json={
                    "text": test_case['text'],
                    "top_k": 5
                }
            )
            data = response.json()
            
            if response.status_code == 200 and data.get('success'):
                print_success(f"Top 5 Predictions:")
                for j, pred in enumerate(data['predictions'], 1):
                    confidence = pred['confidence'] * 100
                    print(f"  {j}. {pred['character']} - {confidence:.1f}% confidence")
                    if j == 1:
                        # Check if top prediction is reasonable
                        if confidence > 20:
                            print_success(f"  → High confidence prediction!")
            else:
                print_error(f"Prediction failed: {data.get('error', 'Unknown error')}")
                all_passed = False
        except Exception as e:
            print_error(f"Prediction failed: {e}")
            all_passed = False
    
    return all_passed

def test_svm_info():
    """Test 4: Get SVM model information"""
    print_header("Test 4: SVM Model Information")
    try:
        response = requests.get(f"{BASE_URL}/svm-info")
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            info = data['model_info']
            print_success("SVM Model Information:")
            print_info(f"Trained: {info['is_trained']}")
            print_info(f"Kernel: {info['kernel']}")
            print_info(f"Number of Classes: {info['n_classes']}")
            print_info(f"Support Vectors: {info['n_support_vectors']}")
            print_info(f"Training Samples: {info['n_training_samples']}")
            print_info(f"Test Samples: {info['n_test_samples']}")
            print_info(f"Training Accuracy: {info['train_accuracy']:.2%}")
            print_info(f"Test Accuracy: {info['test_accuracy']:.2%}")
            print_info(f"TF-IDF Features: {info['n_features']}")
            
            if info['classes']:
                print_info(f"Sample Classes: {', '.join(info['classes'][:5])}...")
            
            return True
        else:
            print_error(f"Failed to get model info: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print_error(f"Failed to get model info: {e}")
        return False

def test_feature_importance():
    """Test 5: Get feature importance (only for linear kernel)"""
    print_header("Test 5: Feature Importance (Linear Kernel)")
    
    print_info("Note: Feature importance only available for linear kernel")
    print_info("Training SVM with linear kernel...")
    
    try:
        # Train with linear kernel
        response = requests.post(
            f"{BASE_URL}/train-svm",
            json={
                "kernel": "linear",
                "optimize": False
            }
        )
        
        if response.status_code != 200 or not response.json().get('success'):
            print_error("Failed to train linear kernel SVM")
            return False
        
        print_success("Linear kernel SVM trained")
        
        # Get feature importance
        response = requests.get(f"{BASE_URL}/svm-feature-importance?top_n=10")
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            print_success("Top 10 Important Features:")
            for i, feature in enumerate(data['features'], 1):
                print(f"  {i}. {feature['feature']} - Weight: {feature['weight']:.4f}")
            return True
        else:
            print_error(f"Failed to get feature importance: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print_error(f"Failed to get feature importance: {e}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"  SVM CHARACTER CLASSIFIER TEST SUITE")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    print_info("Make sure the Flask ML service is running on http://localhost:5000")
    print_info("Start it with: cd ml-service && python app.py\n")
    
    # Run tests
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("SVM Training", test_svm_training()))
    results.append(("SVM Prediction", test_svm_prediction()))
    results.append(("SVM Model Info", test_svm_info()))
    results.append(("Feature Importance", test_feature_importance()))
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n{Fore.CYAN}Results: {passed}/{total} tests passed{Style.RESET_ALL}")
    
    if passed == total:
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"  🎉 ALL TESTS PASSED! SVM IS WORKING PERFECTLY!")
        print(f"{'='*60}{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.YELLOW}{'='*60}")
        print(f"  ⚠ Some tests failed. Check the output above.")
        print(f"{'='*60}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()
