"""
K-Nearest Neighbors Character Prediction Service

This module extracts features from character clues (images + text)
and uses k-NN to predict which character the user is guessing.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import json
import os
from typing import List, Dict, Tuple
import pickle


class CharacterKNN:
    """K-NN model for character prediction based on clues"""
    
    def __init__(self, k: int = 5):
        """
        Initialize the k-NN model
        
        Args:
            k: Number of nearest neighbors to consider
        """
        self.k = k
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast
        self.knn = None
        self.characters = []
        self.feature_vectors = None
        
    def extract_text_features(self, text: str) -> np.ndarray:
        """
        Extract features from text (quotes) using sentence transformers
        
        Args:
            text: Input text (character quote)
            
        Returns:
            Normalized feature vector
        """
        embedding = self.text_model.encode([text])[0]
        return normalize(embedding.reshape(1, -1))[0]
    
    def extract_combined_features(self, quote: str, source: str, 
                                  universe: str, genre: str) -> np.ndarray:
        """
        Extract and combine features from multiple text clues
        
        Args:
            quote: Character quote
            source: Source material (movie/comic)
            universe: Marvel/DC/Other
            genre: Genre description
            
        Returns:
            Combined normalized feature vector
        """
        # Combine all text clues
        combined_text = f"{quote} {source} {universe} {genre}"
        return self.extract_text_features(combined_text)
    
    def train(self, characters_data: List[Dict]) -> None:
        """
        Train the k-NN model on character data
        
        Args:
            characters_data: List of character dictionaries with clues
        """
        print(f"Training k-NN model with {len(characters_data)} characters...")
        
        self.characters = characters_data
        feature_list = []
        
        # Extract features for each character
        for char in characters_data:
            features = self.extract_combined_features(
                quote=char.get('quote', ''),
                source=char.get('source', ''),
                universe=char.get('universe', ''),
                genre=char.get('genre', '')
            )
            feature_list.append(features)
        
        # Stack all features
        self.feature_vectors = np.vstack(feature_list)
        
        # Build k-NN index
        self.knn = NearestNeighbors(
            n_neighbors=min(self.k, len(characters_data)),
            metric='cosine',
            algorithm='brute'
        )
        self.knn.fit(self.feature_vectors)
        
        print(f"✓ k-NN model trained successfully!")
        print(f"  - Feature dimension: {self.feature_vectors.shape[1]}")
        print(f"  - Number of neighbors (k): {self.k}")
    
    def predict(self, quote: str = "", source: str = "", 
                universe: str = "", genre: str = "", 
                top_k: int = None) -> List[Dict]:
        """
        Predict top-k characters based on clues
        
        Args:
            quote: Character quote clue
            source: Source material clue
            universe: Universe clue (Marvel/DC)
            genre: Genre clue
            top_k: Number of top predictions to return (default: self.k)
            
        Returns:
            List of top-k predictions with scores
        """
        if self.knn is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Extract features from query clues
        query_features = self.extract_combined_features(
            quote=quote,
            source=source,
            universe=universe,
            genre=genre
        ).reshape(1, -1)
        
        # Find nearest neighbors
        k = top_k if top_k else self.k
        k = min(k, len(self.characters))
        distances, indices = self.knn.kneighbors(query_features, n_neighbors=k)
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            char = self.characters[idx]
            # Convert cosine distance to similarity score (0-1)
            similarity = 1.0 - dist
            
            results.append({
                'id': char['id'],
                'name': char['name'],
                'universe': char['universe'],
                'confidence': float(similarity),
                'match_score': float(similarity * 100)  # Percentage
            })
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk"""
        model_data = {
            'k': self.k,
            'characters': self.characters,
            'feature_vectors': self.feature_vectors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.k = model_data['k']
        self.characters = model_data['characters']
        self.feature_vectors = model_data['feature_vectors']
        
        # Rebuild k-NN index
        self.knn = NearestNeighbors(
            n_neighbors=min(self.k, len(self.characters)),
            metric='cosine',
            algorithm='brute'
        )
        self.knn.fit(self.feature_vectors)
        
        print(f"✓ Model loaded from {filepath}")
    
    def find_similar_characters(self, character_id: str, top_k: int = 5) -> List[Dict]:
        """
        Find similar characters based on their features using K-NN
        
        Args:
            character_id: The ID of the source character
            top_k: Number of similar characters to return
            
        Returns:
            List of similar characters with similarity scores
        """
        if self.knn is None or len(self.characters) == 0:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find the source character
        source_idx = None
        for idx, char in enumerate(self.characters):
            if char.get('id') == character_id:
                source_idx = idx
                break
        
        if source_idx is None:
            raise ValueError(f"Character with id '{character_id}' not found in training data")
        
        # Get the feature vector of the source character
        source_features = self.feature_vectors[source_idx].reshape(1, -1)
        
        # Find k+1 nearest neighbors (including the character itself)
        distances, indices = self.knn.kneighbors(
            source_features,
            n_neighbors=min(top_k + 1, len(self.characters))
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Skip the source character itself
            if idx == source_idx:
                continue
            
            char = self.characters[idx]
            # Convert cosine distance to similarity score (0-1)
            # Cosine distance = 1 - cosine similarity
            similarity = 1 - dist
            
            results.append({
                'name': char.get('name', 'Unknown'),
                'id': char.get('id', ''),
                'similarity': float(similarity),
                'distance': float(dist),
                'universe': char.get('universe', ''),
                'genre': char.get('genre', ''),
                'quote': char.get('quote', '')
            })
            
            if len(results) >= top_k:
                break
        
        return results


def load_characters_from_json(json_path: str) -> List[Dict]:
    """Load character data from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("K-NN Character Prediction System")
    print("=" * 60)
    
    # This is a test - in production, load from your actual data
    sample_characters = [
        {
            "id": "iron-man",
            "name": "Iron Man",
            "quote": "I am Iron Man.",
            "source": "Iron Man",
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
        }
    ]
    
    # Train model
    knn_model = CharacterKNN(k=5)
    knn_model.train(sample_characters)
    
    # Test prediction
    print("\n" + "=" * 60)
    print("Testing prediction...")
    print("=" * 60)
    
    results = knn_model.predict(
        quote="I am Iron Man",
        universe="Marvel"
    )
    
    print("\nTop predictions:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']} - {result['match_score']:.1f}% confidence")
