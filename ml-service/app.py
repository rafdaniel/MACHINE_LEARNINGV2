"""
Flask API server for Machine Learning Models:
- K-NN Character Prediction
- Linear Regression Difficulty Analysis
- Naive Bayes Genre/Universe Classification
- SVM Character Classification

Provides REST API endpoints that the Express server can call
to get ML-powered character predictions and analysis.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from knn_model import CharacterKNN
from linear_regression_model import CharacterDifficultyPredictor
from naive_bayes_model import CharacterNaiveBayes
from svm_model import CharacterSVM
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Express server

# Initialize global models
knn_model = None
lr_model = None
nb_model = None
svm_model = None


def load_characters_from_typescript():
    """
    Load character data from the TypeScript characters file
    This reads the server/data/characters.ts and extracts character data
    """
    # For now, we'll create a characters.json file manually or via a script
    # In production, you could parse the TypeScript file or export it
    json_path = os.path.join(os.path.dirname(__file__), 'characters.json')
    
    if not os.path.exists(json_path):
        # Return sample data for testing
        return [
            {
                "id": "iron-man",
                "name": "Iron Man",
                "quote": "I am Iron Man.",
                "source": "Iron Man",
                "universe": "Marvel",
                "genre": "Superhero Action"
            },
            {
                "id": "spider-man",
                "name": "Spider-Man",
                "quote": "With great power comes great responsibility.",
                "source": "Spider-Man",
                "universe": "Marvel",
                "genre": "Superhero Action"
            },
            {
                "id": "captain-america",
                "name": "Captain America",
                "quote": "I can do this all day.",
                "source": "Captain America: The First Avenger",
                "universe": "Marvel",
                "genre": "Superhero Action"
            },
            {
                "id": "batman",
                "name": "Batman",
                "quote": "I'm Batman.",
                "source": "Batman Begins",
                "universe": "DC",
                "genre": "Superhero Action"
            },
            {
                "id": "superman",
                "name": "Superman",
                "quote": "I'm here to fight for truth, justice, and the American way.",
                "source": "Superman",
                "universe": "DC",
                "genre": "Superhero Action"
            }
        ]
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'knn': {
                'loaded': knn_model is not None,
                'trained': knn_model is not None and knn_model.knn is not None
            },
            'linear_regression': {
                'loaded': lr_model is not None,
                'trained': lr_model is not None and lr_model.is_trained
            },
            'naive_bayes': {
                'loaded': nb_model is not None,
                'trained': nb_model is not None and nb_model.is_trained
            },
            'svm': {
                'loaded': svm_model is not None,
                'trained': svm_model is not None and svm_model.is_trained
            }
        },
        'service': 'ML Character Analysis (K-NN + LR + Naive Bayes + SVM)'
    })


@app.route('/train', methods=['POST'])
def train_model():
    """
    Train the k-NN model with character data
    
    Body (optional):
        {
            "characters": [...],  // Optional: provide custom training data
            "k": 5                // Optional: number of neighbors
        }
    """
    global knn_model
    
    try:
        data = request.get_json() or {}
        k = data.get('k', 5)
        
        # Load characters
        if 'characters' in data:
            characters = data['characters']
        else:
            characters = load_characters_from_typescript()
        
        # Train model
        knn_model = CharacterKNN(k=k)
        knn_model.train(characters)
        
        return jsonify({
            'success': True,
            'message': f'Model trained with {len(characters)} characters',
            'num_characters': len(characters),
            'k': k
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict_character():
    """
    Predict character based on clues
    
    Body:
        {
            "quote": "optional quote clue",
            "source": "optional source clue",
            "universe": "optional universe clue (Marvel/DC)",
            "genre": "optional genre clue",
            "top_k": 5  // Optional: number of predictions to return
        }
    
    Returns:
        {
            "predictions": [
                {
                    "id": "character-id",
                    "name": "Character Name",
                    "universe": "Marvel",
                    "confidence": 0.95,
                    "match_score": 95.0
                },
                ...
            ]
        }
    """
    global knn_model
    
    if knn_model is None:
        return jsonify({
            'success': False,
            'error': 'Model not trained. Call /train first.'
        }), 400
    
    try:
        data = request.get_json() or {}
        
        # Extract clues
        quote = data.get('quote', '')
        source = data.get('source', '')
        universe = data.get('universe', '')
        genre = data.get('genre', '')
        top_k = data.get('top_k', 5)
        
        # Get predictions
        predictions = knn_model.predict(
            quote=quote,
            source=source,
            universe=universe,
            genre=genre,
            top_k=top_k
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/analyze-clues', methods=['POST'])
def analyze_clues():
    """
    Analyze unlocked clues and return best character matches
    This is a convenience endpoint that takes the game state
    
    Body:
        {
            "clues": {
                "visual": "image_url or null",
                "quote": "quote text or null",
                "source": {
                    "title": "source title",
                    "genre": "genre"
                } or null
            },
            "incorrectGuesses": 3  // Number of incorrect guesses (for weighting)
        }
    """
    global knn_model
    
    if knn_model is None:
        return jsonify({
            'success': False,
            'error': 'Model not trained. Call /train first.'
        }), 400
    
    try:
        data = request.get_json() or {}
        clues = data.get('clues', {})
        
        # Extract available clues
        quote = clues.get('quote', '') if clues.get('quote') else ''
        source_data = clues.get('source', {})
        source = source_data.get('title', '') if source_data else ''
        genre = source_data.get('genre', '') if source_data else ''
        
        # Get predictions
        predictions = knn_model.predict(
            quote=quote,
            source=source,
            genre=genre,
            top_k=10  # Return top 10 for analysis
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'clues_used': {
                'quote': bool(quote),
                'source': bool(source),
                'genre': bool(genre)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/train-lr', methods=['POST'])
def train_linear_regression():
    """
    Train the Linear Regression model for difficulty prediction
    """
    global lr_model
    
    try:
        # Get training data
        data = request.get_json() or {}
        characters = data.get('characters') or load_characters_from_typescript()
        
        # Create and train model
        lr_model = CharacterDifficultyPredictor()
        metrics = lr_model.train(characters)
        
        # Save model
        lr_model.save_model('linear_regression_model.pkl')
        
        return jsonify({
            'success': True,
            'message': 'Linear Regression model trained successfully',
            'metrics': metrics,
            'num_characters': len(characters)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict-difficulty', methods=['POST'])
def predict_difficulty():
    """
    Predict difficulty for a specific character
    
    Body:
        {
            "character": {
                "name": "Spider-Man",
                "attributes": { "powers": [...] },
                ...
            }
        }
    """
    global lr_model
    
    if lr_model is None or not lr_model.is_trained:
        return jsonify({
            'success': False,
            'error': 'Linear Regression model not trained. Call /train-lr first.'
        }), 400
    
    try:
        data = request.get_json()
        character = data.get('character')
        
        if not character:
            return jsonify({
                'success': False,
                'error': 'Character data required'
            }), 400
        
        # Predict difficulty
        prediction = lr_model.predict_difficulty(character)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/difficulty-rankings', methods=['GET'])
def get_difficulty_rankings():
    """
    Get difficulty rankings for all characters
    """
    global lr_model
    
    if lr_model is None or not lr_model.is_trained:
        return jsonify({
            'success': False,
            'error': 'Linear Regression model not trained. Call /train-lr first.'
        }), 400
    
    try:
        # Get all predictions
        rankings = lr_model.predict_all_difficulties()
        
        return jsonify({
            'success': True,
            'rankings': rankings,
            'total_characters': len(rankings)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """
    Get feature importance from Linear Regression model
    """
    global lr_model
    
    if lr_model is None or not lr_model.is_trained:
        return jsonify({
            'success': False,
            'error': 'Linear Regression model not trained. Call /train-lr first.'
        }), 400
    
    try:
        importance = lr_model.get_feature_importance()
        
        return jsonify({
            'success': True,
            'feature_importance': importance
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== NAIVE BAYES ENDPOINTS =====

@app.route('/train-nb', methods=['POST'])
def train_naive_bayes():
    """
    Train the Naive Bayes classifier for genre and universe prediction
    """
    global nb_model
    
    try:
        # Get training data
        data = request.get_json() or {}
        characters = data.get('characters') or load_characters_from_typescript()
        
        # Create and train model
        nb_model = CharacterNaiveBayes()
        metrics = nb_model.train(characters)
        
        # Save model
        nb_model.save_model('naive_bayes_model.pkl')
        
        return jsonify({
            'success': True,
            'message': 'Naive Bayes model trained successfully',
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict-genre', methods=['POST'])
def predict_genre():
    """
    Predict genre based on character text (quote, description, etc.)
    
    Body:
        {
            "text": "Character quote or description",
            "top_k": 3  // optional, default 3
        }
    """
    global nb_model
    
    if nb_model is None or not nb_model.is_trained:
        return jsonify({
            'success': False,
            'error': 'Naive Bayes model not trained. Call /train-nb first.'
        }), 400
    
    try:
        data = request.get_json()
        text = data.get('text')
        top_k = data.get('top_k', 3)
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400
        
        # Predict genre
        predictions = nb_model.predict_genre(text, top_k)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input_text': text
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict-universe', methods=['POST'])
def predict_universe():
    """
    Predict universe based on character text (quote, description, etc.)
    
    Body:
        {
            "text": "Character quote or description",
            "top_k": 3  // optional, default 3
        }
    """
    global nb_model
    
    if nb_model is None or not nb_model.is_trained:
        return jsonify({
            'success': False,
            'error': 'Naive Bayes model not trained. Call /train-nb first.'
        }), 400
    
    try:
        data = request.get_json()
        text = data.get('text')
        top_k = data.get('top_k', 3)
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400
        
        # Predict universe
        predictions = nb_model.predict_universe(text, top_k)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input_text': text
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/classify-character', methods=['POST'])
def classify_character():
    """
    Classify a character (predict both genre and universe)
    
    Body:
        {
            "quote": "Character quote",
            "name": "Character name",
            "source": "Source title",
            "description": "Character description"
        }
    """
    global nb_model
    
    if nb_model is None or not nb_model.is_trained:
        return jsonify({
            'success': False,
            'error': 'Naive Bayes model not trained. Call /train-nb first.'
        }), 400
    
    try:
        character_data = request.get_json()
        
        if not character_data:
            return jsonify({
                'success': False,
                'error': 'Character data is required'
            }), 400
        
        # Classify character
        classification = nb_model.classify_character(character_data)
        
        return jsonify({
            'success': True,
            'classification': classification
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/nb-info', methods=['GET'])
def get_nb_info():
    """
    Get information about the Naive Bayes model
    """
    global nb_model
    
    if nb_model is None:
        return jsonify({
            'success': False,
            'error': 'Naive Bayes model not loaded'
        }), 400
    
    try:
        info = nb_model.get_model_info()
        
        return jsonify({
            'success': True,
            'model_info': info
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== SVM ENDPOINTS =====

@app.route('/train-svm', methods=['POST'])
def train_svm():
    """
    Train the SVM classifier for character identification
    """
    global svm_model
    
    try:
        # Get training data
        data = request.get_json() or {}
        characters = data.get('characters') or load_characters_from_typescript()
        kernel = data.get('kernel', 'rbf')  # linear, rbf, poly, sigmoid
        optimize = data.get('optimize', False)
        
        # Create and train model
        svm_model = CharacterSVM(kernel=kernel, use_calibration=True)
        metrics = svm_model.train(characters, optimize=optimize)
        
        # Save model
        svm_model.save_model('svm_model.pkl')
        
        return jsonify({
            'success': True,
            'message': 'SVM model trained successfully',
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict-svm', methods=['POST'])
def predict_svm():
    """
    Predict character using SVM
    
    Body:
        {
            "text": "Character quote or description",
            "top_k": 5  // optional, default 5
        }
    """
    global svm_model
    
    if svm_model is None or not svm_model.is_trained:
        return jsonify({
            'success': False,
            'error': 'SVM model not trained. Call /train-svm first.'
        }), 400
    
    try:
        data = request.get_json()
        text = data.get('text')
        top_k = data.get('top_k', 5)
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400
        
        # Predict
        predictions = svm_model.predict(text, top_k)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input_text': text
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/svm-feature-importance', methods=['GET'])
def svm_feature_importance():
    """
    Get feature importance from SVM (linear kernel only)
    """
    global svm_model
    
    if svm_model is None or not svm_model.is_trained:
        return jsonify({
            'success': False,
            'error': 'SVM model not trained. Call /train-svm first.'
        }), 400
    
    try:
        top_n = request.args.get('top_n', 20, type=int)
        features = svm_model.get_feature_importance(top_n)
        
        return jsonify({
            'success': True,
            'features': features
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/svm-info', methods=['GET'])
def get_svm_info():
    """
    Get information about the SVM model
    """
    global svm_model
    
    if svm_model is None:
        return jsonify({
            'success': False,
            'error': 'SVM model not loaded'
        }), 400
    
    try:
        info = svm_model.get_model_info()
        
        return jsonify({
            'success': True,
            'model_info': info
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("ML Character Prediction API Server")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nK-NN Endpoints:")
    print("  GET  /health            - Health check")
    print("  POST /train             - Train K-NN model")
    print("  POST /predict           - Predict character from clues")
    print("  POST /analyze-clues     - Analyze game clues")
    print("\nLinear Regression Endpoints:")
    print("  POST /train-lr          - Train Linear Regression model")
    print("  POST /predict-difficulty- Predict character difficulty")
    print("  GET  /difficulty-rankings- Get difficulty rankings")
    print("  GET  /feature-importance - Get feature importance")
    print("\nNaive Bayes Endpoints:")
    print("  POST /train-nb          - Train Naive Bayes classifier")
    print("  POST /predict-genre     - Predict character genre")
    print("  POST /predict-universe  - Predict character universe")
    print("  POST /classify-character- Full classification (genre + universe)")
    print("  GET  /nb-info           - Get Naive Bayes model info")
    print("\nSVM Endpoints:")
    print("  POST /train-svm         - Train SVM classifier")
    print("  POST /predict-svm       - Predict character using SVM")
    print("  GET  /svm-feature-importance - Get feature importance (linear kernel)")
    print("  GET  /svm-info          - Get SVM model info")
    print("=" * 60)
    
    # Auto-train on startup
    try:
        print("\nAuto-training models...")
        characters = load_characters_from_typescript()
        
        # Train K-NN
        knn_model = CharacterKNN(k=5)
        knn_model.train(characters)
        print("✓ K-NN model ready!")
        
        # Train Linear Regression
        lr_model = CharacterDifficultyPredictor()
        lr_metrics = lr_model.train(characters)
        print(f"✓ Linear Regression model ready! (R²={lr_metrics['r2_score']:.4f})")
        
        # Train Naive Bayes
        nb_model = CharacterNaiveBayes()
        nb_metrics = nb_model.train(characters)
        print(f"✓ Naive Bayes model ready! (Genre: {nb_metrics['genre_accuracy']:.2%}, Universe: {nb_metrics['universe_accuracy']:.2%})")
        
        # Train SVM
        svm_model = CharacterSVM(kernel='rbf', use_calibration=True)
        svm_metrics = svm_model.train(characters, optimize=False)
        svm_model.save_model('svm_model.pkl')
        print(f"✓ SVM model ready! (Accuracy: {svm_metrics['test_accuracy']:.2%}, Support Vectors: {svm_metrics['n_support_vectors']}, Kernel: {svm_metrics['kernel']})")
        print()
    except Exception as e:
        print(f"⚠ Could not auto-train models: {e}")
        print("  You'll need to call /train manually\n")
    
    # Start server
    app.run(host='0.0.0.0', port=5000, debug=True)
