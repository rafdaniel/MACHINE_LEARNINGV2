"""
Support Vector Machine (SVM) for Character Classification

This model uses SVM to classify characters based on text features,
providing an alternative to K-NN and Naive Bayes with better performance
on high-dimensional data.
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import json


class CharacterSVM:
    """
    SVM classifier for character identification and classification
    
    Uses Support Vector Machine with TF-IDF features to:
    - Identify characters from quotes/descriptions
    - Classify by genre or universe
    - Provide confidence scores and decision boundaries
    """
    
    def __init__(self, kernel='rbf', use_calibration=True):
        """
        Initialize SVM classifier
        
        Args:
            kernel: 'linear', 'rbf', 'poly', or 'sigmoid'
            use_calibration: If True, wrap SVC in CalibratedClassifierCV for probabilities
        """
        self.kernel = kernel
        self.use_calibration = use_calibration
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.scaler = StandardScaler(with_mean=False)  # sparse-safe scaler
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        self.training_info = None
        
    def prepare_features(self, characters):
        """
        Extract text features from character data
        
        Args:
            characters: List of character dictionaries
            
        Returns:
            features: Combined text features
            labels: Character names or classes
        """
        features = []
        labels = []
        
        for char in characters:
            # Combine text features
            text_parts = []
            
            if char.get('quote'):
                text_parts.append(char['quote'])
            if char.get('source'):
                text_parts.append(char['source'])
            if char.get('name'):
                text_parts.append(char['name'])
            if char.get('description'):
                text_parts.append(char['description'])
            if char.get('genre'):
                text_parts.append(char['genre'])
            if char.get('universe'):
                text_parts.append(char['universe'])
            
            combined_text = ' '.join(text_parts)
            features.append(combined_text)
            labels.append(char.get('name', 'Unknown'))
        
        return features, labels
    
    def train(self, characters, optimize=False):
        """
        Train the SVM classifier
        
        Args:
            characters: List of character dictionaries
            optimize: If True, use GridSearchCV to find best parameters
            
        Returns:
            metrics: Dictionary containing training metrics
        """
        if len(characters) < 2:
            raise ValueError("Need at least 2 characters to train")
        
        # Prepare features
        features, labels = self.prepare_features(characters)
        
        # Vectorize text
        X = self.vectorizer.fit_transform(features)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split for evaluation
        # Check if we can do stratified split (need at least 2 samples per class)
        unique_labels, label_counts = np.unique(y, return_counts=True)
        can_stratify = all(count >= 2 for count in label_counts)
        
        if can_stratify and len(characters) > 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # Not enough data for stratified split or too few samples
            # Use all data for training
            X_train, X_test = X, X
            y_train, y_test = y, y
            print("Warning: Using all data for training (too few samples for proper split)")
        
        # Create SVM model
        if self.kernel == 'linear':
            # LinearSVC is faster for linear kernel
            base_model = LinearSVC(C=1.0, max_iter=5000, dual='auto')
            if self.use_calibration:
                self.model = CalibratedClassifierCV(base_model, cv=3)
            else:
                self.model = base_model
        else:
            # SVC for non-linear kernels
            base_model = SVC(
                kernel=self.kernel,
                C=1.0,
                gamma='scale',
                probability=self.use_calibration
            )
            self.model = base_model
        
        # Optimize hyperparameters if requested
        if optimize and len(characters) > 10:
            print("Optimizing SVM hyperparameters...")
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto'] if self.kernel != 'linear' else ['scale']
            }
            
            if self.use_calibration and self.kernel == 'linear':
                param_grid = {'base_estimator__C': param_grid['C']}
            
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=min(3, len(np.unique(y))),
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Train with default parameters
            self.model.fit(X_train, y_train)
            best_params = None
        
        # Evaluate
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Get support vector info (if available)
        n_support_vectors = None
        if hasattr(self.model, 'n_support_'):
            n_support_vectors = int(np.sum(self.model.n_support_))
        elif hasattr(self.model, 'base_estimator') and hasattr(self.model.base_estimator, 'n_support_'):
            n_support_vectors = int(np.sum(self.model.base_estimator.n_support_))
        
        self.is_trained = True
        self.training_info = {
            'num_characters': len(characters),
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'kernel': self.kernel,
            'calibrated': self.use_calibration,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'n_support_vectors': n_support_vectors
        }
        
        return {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'num_samples': len(characters),
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'kernel': self.kernel,
            'calibrated': self.use_calibration,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'n_support_vectors': n_support_vectors,
            'best_params': best_params
        }
    
    def predict(self, text, top_k=5):
        """
        Predict character from text input
        
        Args:
            text: Input text (quote, description, etc.)
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Vectorize input
        X = self.vectorizer.transform([text])
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            # Calibrated model with probabilities
            probabilities = self.model.predict_proba(X)[0]
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            
            predictions = []
            for idx in top_indices:
                predictions.append({
                    'character': self.label_encoder.classes_[idx],
                    'confidence': float(probabilities[idx]),
                    'confidence_pct': f"{probabilities[idx] * 100:.1f}%"
                })
        else:
            # Non-calibrated model - use decision function
            decision_scores = self.model.decision_function(X)[0]
            
            # Normalize scores to 0-1 range
            if len(decision_scores.shape) == 0 or decision_scores.shape[0] == 1:
                # Binary classification
                score = float(decision_scores)
                normalized_score = 1 / (1 + np.exp(-score))  # sigmoid
                predictions = [{
                    'character': self.label_encoder.classes_[1 if score > 0 else 0],
                    'confidence': normalized_score if score > 0 else 1 - normalized_score,
                    'confidence_pct': f"{(normalized_score if score > 0 else 1 - normalized_score) * 100:.1f}%",
                    'decision_score': score
                }]
            else:
                # Multi-class
                top_indices = np.argsort(decision_scores)[::-1][:top_k]
                
                predictions = []
                for idx in top_indices:
                    score = float(decision_scores[idx])
                    predictions.append({
                        'character': self.label_encoder.classes_[idx],
                        'decision_score': score,
                        'confidence_pct': f"Score: {score:.3f}"
                    })
        
        return predictions
    
    def predict_batch(self, texts):
        """
        Predict for multiple texts
        
        Args:
            texts: List of text inputs
            
        Returns:
            List of predictions for each text
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.vectorizer.transform(texts)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            predictions = self.label_encoder.inverse_transform(np.argmax(probabilities, axis=1))
            confidences = np.max(probabilities, axis=1)
            
            return [
                {
                    'character': pred,
                    'confidence': float(conf),
                    'confidence_pct': f"{conf * 100:.1f}%"
                }
                for pred, conf in zip(predictions, confidences)
            ]
        else:
            predictions = self.model.predict(X)
            decoded = self.label_encoder.inverse_transform(predictions)
            return [{'character': pred, 'confidence': 1.0} for pred in decoded]
    
    def get_feature_importance(self, top_n=20):
        """
        Get most important features (for linear kernel only)
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of feature names and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Only works for linear models
        coef = None
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
        elif hasattr(self.model, 'base_estimator') and hasattr(self.model.base_estimator, 'coef_'):
            coef = self.model.base_estimator.coef_
        
        if coef is None:
            return {'error': 'Feature importance only available for linear kernel'}
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Average importance across all classes
        importance = np.abs(coef).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(importance)[::-1][:top_n]
        
        features = []
        for idx in top_indices:
            features.append({
                'feature': feature_names[idx],
                'importance': float(importance[idx])
            })
        
        return features
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return {'trained': False}
        
        info = {
            'trained': True,
            'training_info': self.training_info,
            'model_type': 'Support Vector Machine (SVM)',
            'kernel': self.kernel,
            'has_probabilities': hasattr(self.model, 'predict_proba')
        }
        
        return info
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'training_info': self.training_info,
            'kernel': self.kernel,
            'use_calibration': self.use_calibration
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.training_info = model_data['training_info']
        self.kernel = model_data['kernel']
        self.use_calibration = model_data['use_calibration']
        self.is_trained = True


if __name__ == "__main__":
    # Test the SVM model
    print("Testing SVM Character Classifier...")
    
    # Sample data
    sample_characters = [
        {
            "name": "Iron Man",
            "quote": "I am Iron Man",
            "source": "Iron Man",
            "genre": "Superhero Action",
            "universe": "Marvel"
        },
        {
            "name": "Spider-Man",
            "quote": "With great power comes great responsibility",
            "source": "Spider-Man",
            "genre": "Superhero Action",
            "universe": "Marvel"
        },
        {
            "name": "Batman",
            "quote": "I'm Batman",
            "source": "The Dark Knight",
            "genre": "Superhero Action",
            "universe": "DC"
        },
        {
            "name": "Superman",
            "quote": "Truth, justice, and the American way",
            "source": "Superman",
            "genre": "Superhero Action",
            "universe": "DC"
        },
        {
            "name": "Luke Skywalker",
            "quote": "I am a Jedi, like my father before me",
            "source": "Return of the Jedi",
            "genre": "Sci-Fi Action",
            "universe": "Star Wars"
        }
    ]
    
    # Train model
    model = CharacterSVM(kernel='rbf', use_calibration=True)
    metrics = model.train(sample_characters)
    
    print("\nTraining Results:")
    print(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Kernel: {metrics['kernel']}")
    print(f"Support Vectors: {metrics['n_support_vectors']}")
    print(f"Classes: {metrics['classes']}")
    
    # Test prediction
    test_quote = "With great power"
    print(f"\nTesting with quote: '{test_quote}'")
    predictions = model.predict(test_quote, top_k=3)
    
    print("\nTop 3 Predictions:")
    for pred in predictions:
        print(f"  {pred['character']}: {pred.get('confidence_pct', pred.get('decision_score', 'N/A'))}")
