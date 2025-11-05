"""
Test script for Artificial Neural Network (ANN) Character Classifier & Regressor
Tests all ANN endpoints to verify functionality
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
    """Test 1: Check ML service health including ANN"""
    print_header("Test 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        if response.status_code == 200:
            print_success(f"Service Status: {data['status']}")
            print_success(f"Service Name: {data['service']}")
            
            models = data['models']
            print_info(f"K-NN: {'✓ Loaded' if models['knn']['loaded'] else '✗ Not loaded'}, {'✓ Trained' if models['knn']['trained'] else '✗ Not trained'}")
            print_info(f"Linear Regression: {'✓ Loaded' if models['linear_regression']['loaded'] else '✗ Not loaded'}, {'✓ Trained' if models['linear_regression']['trained'] else '✗ Not trained'}")
            print_info(f"Naive Bayes: {'✓ Loaded' if models['naive_bayes']['loaded'] else '✗ Not loaded'}, {'✓ Trained' if models['naive_bayes']['trained'] else '✗ Not trained'}")
            print_info(f"SVM: {'✓ Loaded' if models['svm']['loaded'] else '✗ Not loaded'}, {'✓ Trained' if models['svm']['trained'] else '✗ Not trained'}")
            
            dt_model = models['decision_tree']
            print_info(f"Decision Tree: {'✓ Loaded' if dt_model['loaded'] else '✗ Not loaded'}")
            print_info(f"  - Classifier: {'✓ Trained' if dt_model['trained_classifier'] else '✗ Not trained'}")
            print_info(f"  - Regressor: {'✓ Trained' if dt_model['trained_regressor'] else '✗ Not trained'}")
            
            ann_model = models['ann']
            print_info(f"ANN: {'✓ Loaded' if ann_model['loaded'] else '✗ Not loaded'}")
            print_info(f"  - Classifier: {'✓ Trained' if ann_model['trained_classifier'] else '✗ Not trained'}")
            print_info(f"  - Regressor: {'✓ Trained' if ann_model['trained_regressor'] else '✗ Not trained'}")
            
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_ann_training():
    """Test 2: Train ANN model"""
    print_header("Test 2: Train ANN Model")
    try:
        print_info("Training ANN with architecture [256, 128, 64]...")
        print_info("This may take 1-3 minutes (generating embeddings + training)...")
        
        response = requests.post(
            f"{BASE_URL}/train-ann",
            json={
                "hidden_layers": [256, 128, 64],
                "max_iter": 300,
                "learning_rate": 0.001
            }
        )
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            print_success("ANN models trained successfully!")
            
            metrics = data['metrics']
            
            # Classifier metrics
            print_info("\nClassifier Metrics:")
            clf = metrics['classifier']
            print_info(f"  Training Accuracy: {clf['train_accuracy']:.2%}")
            print_info(f"  Test Accuracy: {clf['test_accuracy']:.2%}")
            print_info(f"  Training Loss: {clf['train_loss']:.4f}")
            if clf.get('val_loss'):
                print_info(f"  Validation Loss: {clf['val_loss']:.4f}")
            print_info(f"  Number of Classes: {clf['n_classes']}")
            print_info(f"  Number of Features: {clf['n_features']}")
            print_info(f"  Training Iterations: {clf['n_iterations']}")
            print_info(f"  Hidden Layers: {clf['hidden_layers']}")
            
            # Regressor metrics
            print_info("\nRegressor Metrics:")
            reg = metrics['regressor']
            print_info(f"  Training R²: {reg['train_r2']:.4f}")
            print_info(f"  Test R²: {reg['test_r2']:.4f}")
            print_info(f"  Training Loss: {reg['train_loss']:.4f}")
            print_info(f"  Training Iterations: {reg['n_iterations']}")
            
            print_info(f"\nTraining Samples: {metrics['n_training_samples']}")
            print_info(f"Test Samples: {metrics['n_test_samples']}")
            
            return True
        else:
            print_error(f"Training failed: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print_error(f"Training failed: {e}")
        return False

def test_ann_prediction():
    """Test 3: Predict character using ANN"""
    print_header("Test 3: ANN Character Prediction")
    
    # Test cases
    test_cases = [
        {
            "character": {
                "name": "Spider-Man",
                "quote": "With great power comes great responsibility",
                "universe": "Marvel",
                "genre": "Superhero Action",
                "powers": ["web-slinging", "spider-sense", "wall-crawling"],
                "description": "A young hero with spider powers",
                "source": "Spider-Man"
            },
            "description": "Spider-Man test"
        },
        {
            "character": {
                "name": "Batman",
                "quote": "I am vengeance, I am the night",
                "universe": "DC",
                "genre": "Superhero Action",
                "powers": ["martial arts", "detective skills", "gadgets"],
                "description": "Dark knight of Gotham",
                "source": "Batman"
            },
            "description": "Batman test"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{Fore.YELLOW}Test Case {i}: {test_case['description']}{Style.RESET_ALL}")
        print(f"Character: {test_case['character']['name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict-ann",
                json={
                    "character": test_case['character'],
                    "top_k": 5
                }
            )
            data = response.json()
            
            if response.status_code == 200 and data.get('success'):
                print_success(f"Top 5 Predictions:")
                for j, pred in enumerate(data['predictions'], 1):
                    prob = pred['probability'] * 100
                    print(f"  {j}. {pred['character']} - {prob:.1f}%")
            else:
                print_error(f"Prediction failed: {data.get('error', 'Unknown error')}")
                all_passed = False
        except Exception as e:
            print_error(f"Prediction failed: {e}")
            all_passed = False
    
    return all_passed

def test_ann_difficulty():
    """Test 4: Predict difficulty using ANN regressor"""
    print_header("Test 4: ANN Difficulty Prediction")
    
    test_character = {
        "name": "Thor",
        "quote": "I am Thor, son of Odin",
        "universe": "Marvel",
        "genre": "Superhero Action",
        "powers": ["super strength", "lightning", "mjolnir"],
        "description": "God of Thunder",
        "source": "Thor"
    }
    
    try:
        print_info(f"Predicting difficulty for: {test_character['name']}")
        
        response = requests.post(
            f"{BASE_URL}/predict-difficulty-ann",
            json={"character": test_character}
        )
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            print_success(f"Predicted Difficulty: {data['difficulty']:.1f}/10")
            return True
        else:
            print_error(f"Prediction failed: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print_error(f"Prediction failed: {e}")
        return False

def test_ann_info():
    """Test 5: Get ANN model information"""
    print_header("Test 5: ANN Model Information")
    try:
        response = requests.get(f"{BASE_URL}/ann-info")
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            info = data['model_info']
            
            print_success("Classifier Information:")
            clf = info['classifier']
            print_info(f"  Trained: {clf['is_trained']}")
            print_info(f"  Number of Classes: {clf['n_classes']}")
            print_info(f"  Training Accuracy: {clf['train_accuracy']:.2%}")
            print_info(f"  Test Accuracy: {clf['test_accuracy']:.2%}")
            print_info(f"  Training Loss: {clf['train_loss']:.4f}")
            if clf.get('val_loss'):
                print_info(f"  Validation Loss: {clf['val_loss']:.4f}")
            print_info(f"  Training Iterations: {clf['n_iterations']}")
            print_info(f"  Hidden Layers: {clf['hidden_layers']}")
            print_info(f"  Activation: {clf['activation']}")
            print_info(f"  Optimizer: {clf['solver']}")
            
            print_success("\nRegressor Information:")
            reg = info['regressor']
            print_info(f"  Trained: {reg['is_trained']}")
            print_info(f"  Training R²: {reg['train_r2']:.4f}")
            print_info(f"  Test R²: {reg['test_r2']:.4f}")
            print_info(f"  Training Loss: {reg['train_loss']:.4f}")
            print_info(f"  Training Iterations: {reg['n_iterations']}")
            print_info(f"  Hidden Layers: {reg['hidden_layers']}")
            
            print_success("\nArchitecture:")
            arch = info['architecture']
            print_info(f"  Input Features: {arch['input_features']}")
            print_info(f"  Embedding Model: {arch['embedding_model']}")
            print_info(f"  Optimizer: {arch['optimizer']}")
            print_info(f"  Regularization: {arch['regularization']}")
            print_info(f"  Early Stopping: {arch['early_stopping']}")
            
            return True
        else:
            print_error(f"Failed to get model info: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print_error(f"Failed to get model info: {e}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"  ARTIFICIAL NEURAL NETWORK (ANN) TEST SUITE")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    print_info("Make sure the Flask ML service is running on http://localhost:5000")
    print_info("Start it with: cd ml-service && python app.py\n")
    print_info("⚠️  WARNING: ANN training takes longer than other models (1-3 minutes)")
    print_info("    This is normal - it's generating 384-dim embeddings and training deep layers\n")
    
    # Run tests
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("ANN Training", test_ann_training()))
    results.append(("Character Prediction", test_ann_prediction()))
    results.append(("Difficulty Prediction", test_ann_difficulty()))
    results.append(("Model Information", test_ann_info()))
    
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
        print(f"  🎉 ALL TESTS PASSED! ANN IS WORKING PERFECTLY!")
        print(f"{'='*60}{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.YELLOW}{'='*60}")
        print(f"  ⚠ Some tests failed. Check the output above.")
        print(f"{'='*60}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()
