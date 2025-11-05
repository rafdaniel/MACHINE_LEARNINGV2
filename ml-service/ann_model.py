"""
Artificial Neural Network (ANN) Model for Character Classification and Difficulty Prediction

This module provides two MLP (Multi-Layer Perceptron) neural network models:
1. Character Classification: Predict character ID using embeddings + engineered features
2. Difficulty Regression: Predict character difficulty (0-10 scale)

Key Features:
- Deep learning with multiple hidden layers
- Combines sentence embeddings (384-dim) with engineered features
- Handles non-linear relationships in data
- Adaptive learning with Adam optimizer
- Early stopping to prevent overfitting
"""

import numpy as np
import json
import pickle
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


class CharacterANN:
    """
    Artificial Neural Network model for character classification and difficulty prediction
    
    Uses a combination of:
    - Sentence embeddings (384 dimensions) from quotes/descriptions
    - Engineered numerical features (powers_count, name_length, etc.)
    - Categorical features (universe, genre) encoded
    
    Architecture:
    - Input layer: 384 (embeddings) + ~10 (engineered features)
    - Hidden layers: Configurable (default: [256, 128, 64])
    - Output layer: N classes (classifier) or 1 (regressor)
    """
    
    def __init__(self, hidden_layers=(256, 128, 64), max_iter=300, learning_rate=0.001):
        """
        Initialize ANN models
        
        Args:
            hidden_layers: Tuple of hidden layer sizes (e.g., (256, 128, 64))
            max_iter: Maximum training iterations/epochs
            learning_rate: Initial learning rate for Adam optimizer
        """
        # Classifier for character prediction
        self.classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
        
        # Regressor for difficulty prediction
        self.regressor = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.universe_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        
        # Sentence transformer for text embeddings
        self.text_model = None
        
        self.is_trained_classifier = False
        self.is_trained_regressor = False
        self.class_names = []
        
        # Training metrics
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.train_r2 = 0.0
        self.test_r2 = 0.0
        self.n_iterations = 0
        
    def _load_text_model(self):
        """Load sentence transformer model (lazy loading)"""
        if self.text_model is None:
            print("Loading sentence transformer model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _extract_features(self, characters, fit=False):
        """
        Extract and combine features from character data
        
        Returns:
            X: Feature matrix (embeddings + engineered features)
            y_cls: Character IDs for classification
            y_reg: Difficulty scores for regression
        """
        self._load_text_model()
        
        # Collect text for embeddings
        texts = []
        for char in characters:
            quote = char.get('quote', '') or ''
            name = char.get('name', '') or ''
            description = char.get('description', '') or ''
            source = char.get('source', '') or ''
            text = f"{quote} {name} {description} {source}"
            texts.append(text)
        
        # Generate sentence embeddings (384 dimensions)
        print(f"Generating embeddings for {len(texts)} characters...")
        embeddings = self.text_model.encode(texts, show_progress_bar=False)
        
        # Engineered features
        engineered_features = []
        y_cls = []
        y_reg = []
        universes = []
        genres = []
        
        for char in characters:
            # Numerical features
            powers_count = len(char.get('powers', []))
            name_length = len(char.get('name', '') or '')
            quote_length = len(char.get('quote', '') or '')
            description_length = len(char.get('description', '') or '')
            source_length = len(char.get('source', '') or '')
            
            # Categorical features
            universe = char.get('universe', 'Unknown')
            genre = char.get('genre', 'Unknown')
            universes.append(universe)
            genres.append(genre)
            
            # Combine numerical features
            feature_vector = [
                powers_count,
                name_length,
                quote_length,
                description_length,
                source_length
            ]
            
            engineered_features.append(feature_vector)
            y_cls.append(char['id'])
            y_reg.append(char.get('difficulty', 5))
        
        engineered_features = np.array(engineered_features)
        
        # Encode categorical features
        if fit:
            universe_encoded = self.universe_encoder.fit_transform(universes).reshape(-1, 1)
            genre_encoded = self.genre_encoder.fit_transform(genres).reshape(-1, 1)
        else:
            universe_encoded = self.universe_encoder.transform(universes).reshape(-1, 1)
            genre_encoded = self.genre_encoder.transform(genres).reshape(-1, 1)
        
        # Combine all features: embeddings + numerical + categorical
        X = np.concatenate([
            embeddings,
            engineered_features,
            universe_encoded,
            genre_encoded
        ], axis=1)
        
        return X, y_cls, y_reg
    
    def train(self, characters):
        """
        Train both classifier and regressor models
        
        Args:
            characters: List of character dictionaries
            
        Returns:
            metrics: Dictionary with training metrics
        """
        print(f"\nTraining Artificial Neural Network with {len(characters)} characters...")
        
        # Extract features
        X, y_cls, y_reg = self._extract_features(characters, fit=True)
        
        print(f"Feature dimensions: {X.shape[1]} (384 embeddings + {X.shape[1] - 384} engineered)")
        
        # Encode classification labels
        y_cls_encoded = self.label_encoder.fit_transform(y_cls)
        self.class_names = self.label_encoder.classes_.tolist()
        
        # Check if we can do train-test split
        unique_labels, label_counts = np.unique(y_cls_encoded, return_counts=True)
        can_split = all(count >= 2 for count in label_counts) and len(characters) > 5
        
        if can_split:
            # Split data
            X_train, X_test, ytrain_cls, ytest_cls, ytrain_reg, ytest_reg = train_test_split(
                X, y_cls_encoded, y_reg, test_size=0.2, random_state=42, stratify=y_cls_encoded
            )
        else:
            # Use all data for training
            X_train = X_test = X
            ytrain_cls = ytest_cls = y_cls_encoded
            ytrain_reg = ytest_reg = y_reg
            print("Warning: Using all data for training (too few samples for proper split)")
        
        # Scale features (important for neural networks)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        print("Training neural network classifier...")
        self.classifier.fit(X_train_scaled, ytrain_cls)
        self.train_accuracy = self.classifier.score(X_train_scaled, ytrain_cls)
        self.test_accuracy = self.classifier.score(X_test_scaled, ytest_cls)
        self.n_iterations = self.classifier.n_iter_
        self.train_loss = self.classifier.loss_
        self.val_loss = self.classifier.best_loss_ if hasattr(self.classifier, 'best_loss_') else None
        self.is_trained_classifier = True
        
        print(f"✓ ANN Classifier trained! (Accuracy: {self.test_accuracy:.2%}, Iterations: {self.n_iterations})")
        
        # Train regressor
        print("Training neural network regressor...")
        self.regressor.fit(X_train_scaled, ytrain_reg)
        self.train_r2 = self.regressor.score(X_train_scaled, ytrain_reg)
        self.test_r2 = self.regressor.score(X_test_scaled, ytest_reg)
        self.is_trained_regressor = True
        
        print(f"✓ ANN Regressor trained! (R²: {self.test_r2:.4f})")
        
        return {
            'classifier': {
                'train_accuracy': float(self.train_accuracy),
                'test_accuracy': float(self.test_accuracy),
                'train_loss': float(self.train_loss),
                'val_loss': float(self.val_loss) if self.val_loss else None,
                'n_classes': len(self.class_names),
                'n_features': X.shape[1],
                'n_iterations': int(self.n_iterations),
                'hidden_layers': list(self.classifier.hidden_layer_sizes)
            },
            'regressor': {
                'train_r2': float(self.train_r2),
                'test_r2': float(self.test_r2),
                'train_loss': float(self.regressor.loss_),
                'n_iterations': int(self.regressor.n_iter_)
            },
            'n_training_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
    
    def predict_character(self, character_data, top_k=5):
        """
        Predict character ID from character data
        
        Args:
            character_data: Character dictionary with features
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with probabilities
        """
        if not self.is_trained_classifier:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Extract features
        X, _, _ = self._extract_features([character_data], fit=False)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probas = self.classifier.predict_proba(X_scaled)[0]
        
        # Get top k predictions
        top_indices = np.argsort(probas)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            if probas[idx] > 0:
                predictions.append({
                    'character': self.label_encoder.inverse_transform([idx])[0],
                    'probability': float(probas[idx]),
                    'confidence': float(probas[idx] * 100)
                })
        
        return predictions
    
    def predict_difficulty(self, character_data):
        """
        Predict difficulty score for a character
        
        Args:
            character_data: Character dictionary with features
            
        Returns:
            Predicted difficulty (0-10)
        """
        if not self.is_trained_regressor:
            raise ValueError("Regressor not trained. Call train() first.")
        
        # Extract features
        X, _, _ = self._extract_features([character_data], fit=False)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        difficulty = self.regressor.predict(X_scaled)[0]
        
        # Clip to valid range
        difficulty = np.clip(difficulty, 0, 10)
        
        return float(difficulty)
    
    def get_model_info(self):
        """
        Get information about trained models
        
        Returns:
            Dictionary with model information
        """
        info = {
            'classifier': {
                'is_trained': self.is_trained_classifier,
                'n_classes': len(self.class_names) if self.is_trained_classifier else 0,
                'classes': self.class_names if self.is_trained_classifier else [],
                'train_accuracy': float(self.train_accuracy) if self.is_trained_classifier else 0,
                'test_accuracy': float(self.test_accuracy) if self.is_trained_classifier else 0,
                'train_loss': float(self.train_loss) if self.is_trained_classifier else 0,
                'val_loss': float(self.val_loss) if self.val_loss and self.is_trained_classifier else None,
                'n_iterations': int(self.n_iterations) if self.is_trained_classifier else 0,
                'hidden_layers': list(self.classifier.hidden_layer_sizes) if self.is_trained_classifier else [],
                'activation': 'relu',
                'solver': 'adam'
            },
            'regressor': {
                'is_trained': self.is_trained_regressor,
                'train_r2': float(self.train_r2) if self.is_trained_regressor else 0,
                'test_r2': float(self.test_r2) if self.is_trained_regressor else 0,
                'train_loss': float(self.regressor.loss_) if self.is_trained_regressor else 0,
                'n_iterations': int(self.regressor.n_iter_) if self.is_trained_regressor else 0,
                'hidden_layers': list(self.regressor.hidden_layer_sizes) if self.is_trained_regressor else []
            },
            'architecture': {
                'input_features': '384 (embeddings) + ~7 (engineered)',
                'embedding_model': 'all-MiniLM-L6-v2',
                'optimizer': 'adam',
                'regularization': 'L2 (alpha=0.0001)',
                'early_stopping': True
            }
        }
        
        return info
    
    def save_model(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'regressor': self.regressor,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'universe_encoder': self.universe_encoder,
                'genre_encoder': self.genre_encoder,
                'class_names': self.class_names,
                'is_trained_classifier': self.is_trained_classifier,
                'is_trained_regressor': self.is_trained_regressor,
                'train_accuracy': self.train_accuracy,
                'test_accuracy': self.test_accuracy,
                'train_loss': self.train_loss,
                'val_loss': self.val_loss,
                'train_r2': self.train_r2,
                'test_r2': self.test_r2,
                'n_iterations': self.n_iterations
            }, f)
    
    def load_model(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.regressor = data['regressor']
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.universe_encoder = data['universe_encoder']
            self.genre_encoder = data['genre_encoder']
            self.class_names = data['class_names']
            self.is_trained_classifier = data['is_trained_classifier']
            self.is_trained_regressor = data['is_trained_regressor']
            self.train_accuracy = data['train_accuracy']
            self.test_accuracy = data['test_accuracy']
            self.train_loss = data['train_loss']
            self.val_loss = data['val_loss']
            self.train_r2 = data['train_r2']
            self.test_r2 = data['test_r2']
            self.n_iterations = data['n_iterations']
        
        # Reload text model
        self._load_text_model()


# Test code
if __name__ == "__main__":
    # Sample test data
    test_characters = [
        {
            "id": "spider-man",
            "name": "Spider-Man",
            "quote": "With great power comes great responsibility.",
            "source": "Spider-Man",
            "universe": "Marvel",
            "genre": "Superhero Action",
            "powers": ["web-slinging", "wall-crawling", "spider-sense"],
            "difficulty": 7,
            "description": "A young hero with spider powers"
        },
        {
            "id": "iron-man",
            "name": "Iron Man",
            "quote": "I am Iron Man.",
            "source": "Iron Man",
            "universe": "Marvel",
            "genre": "Superhero Action",
            "powers": ["powered armor", "genius intellect", "flight"],
            "difficulty": 6,
            "description": "Genius billionaire in powered armor"
        },
        {
            "id": "batman",
            "name": "Batman",
            "quote": "I'm Batman.",
            "source": "Batman",
            "universe": "DC",
            "genre": "Superhero Action",
            "powers": ["martial arts", "detective skills", "gadgets"],
            "difficulty": 8,
            "description": "Dark knight detective of Gotham"
        }
    ]
    
    print("Testing Artificial Neural Network Model...")
    
    # Initialize and train
    ann_model = CharacterANN(hidden_layers=(128, 64), max_iter=200)
    metrics = ann_model.train(test_characters)
    
    print("\nTraining Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Test character prediction
    print("\nTesting character prediction...")
    test_char = {
        "quote": "With great power",
        "name": "Spider-Man",
        "universe": "Marvel",
        "genre": "Superhero Action",
        "powers": ["web-slinging", "spider-sense"],
        "description": "A hero with spider abilities",
        "source": "Spider-Man"
    }
    
    predictions = ann_model.predict_character(test_char, top_k=3)
    print(f"\nTop 3 predictions:")
    for pred in predictions:
        print(f"  {pred['character']}: {pred['confidence']:.1f}%")
    
    # Test difficulty prediction
    difficulty = ann_model.predict_difficulty(test_char)
    print(f"\nPredicted difficulty: {difficulty:.1f}/10")
    
    # Model info
    info = ann_model.get_model_info()
    print(f"\nModel Architecture:")
    print(f"  Classifier Layers: {info['classifier']['hidden_layers']}")
    print(f"  Training Iterations: {info['classifier']['n_iterations']}")
    print(f"  Activation: {info['classifier']['activation']}")
    print(f"  Optimizer: {info['classifier']['solver']}")
    
    print("\n✓ Artificial Neural Network model test complete!")
