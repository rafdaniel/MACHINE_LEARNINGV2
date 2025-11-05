import { RequestHandler } from "express";

/**
 * POST /api/ml/predict
 * Get ML-powered character predictions based on clues
 * 
 * Body:
 *   {
 *     "quote": "optional quote",
 *     "source": "optional source",
 *     "universe": "optional universe",
 *     "genre": "optional genre",
 *     "top_k": 5
 *   }
 */
export const handleMLPredict: RequestHandler = async (req, res) => {
  try {
    const { quote = "", source = "", universe = "", genre = "", top_k = 5 } = req.body;
    
    // Call Python ML service
    const mlResponse = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quote, source, universe, genre, top_k })
    });
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML prediction error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable. Make sure Python service is running on port 5000.' 
    });
  }
};

/**
 * POST /api/ml/analyze-game
 * Analyze current game state and get ML suggestions
 * 
 * Body:
 *   {
 *     "clues": { visual, quote, source },
 *     "incorrectGuesses": number
 *   }
 */
export const handleMLAnalyzeGame: RequestHandler = async (req, res) => {
  try {
    const { clues, incorrectGuesses = 0 } = req.body;
    
    // Call Python ML service
    const mlResponse = await fetch('http://localhost:5000/analyze-clues', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clues, incorrectGuesses })
    });
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML analysis error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

/**
 * GET /api/ml/health
 * Check if ML service is available
 */
export const handleMLHealth: RequestHandler = async (_req, res) => {
  try {
    const mlResponse = await fetch('http://localhost:5000/health');
    const data = await mlResponse.json();
    res.json(data);
  } catch (error) {
    res.status(503).json({ 
      status: 'unavailable',
      error: 'ML service is not running'
    });
  }
};

/**
 * POST /api/ml/predict-difficulty
 * Predict character difficulty using Linear Regression
 * 
 * Body:
 *   {
 *     "character": { name, attributes, etc. }
 *   }
 */
export const handleMLPredictDifficulty: RequestHandler = async (req, res) => {
  try {
    const { character } = req.body;
    
    if (!character) {
      return res.status(400).json({
        success: false,
        error: 'Character data required'
      });
    }
    
    // Call Python ML service
    const mlResponse = await fetch('http://localhost:5000/predict-difficulty', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ character })
    });
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML difficulty prediction error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

/**
 * GET /api/ml/difficulty-rankings
 * Get difficulty rankings for all characters
 */
export const handleMLDifficultyRankings: RequestHandler = async (_req, res) => {
  try {
    const mlResponse = await fetch('http://localhost:5000/difficulty-rankings');
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML difficulty rankings error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

/**
 * GET /api/ml/feature-importance
 * Get feature importance from Linear Regression model
 */
export const handleMLFeatureImportance: RequestHandler = async (_req, res) => {
  try {
    const mlResponse = await fetch('http://localhost:5000/feature-importance');
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML feature importance error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

// ===== NAIVE BAYES ENDPOINTS =====

export const handleMLPredictGenre: RequestHandler = async (req, res) => {
  try {
    const { text, top_k } = req.body;
    
    const mlResponse = await fetch('http://localhost:5000/predict-genre', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, top_k })
    });
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML genre prediction error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

export const handleMLPredictUniverse: RequestHandler = async (req, res) => {
  try {
    const { text, top_k } = req.body;
    
    const mlResponse = await fetch('http://localhost:5000/predict-universe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, top_k })
    });
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML universe prediction error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

export const handleMLClassifyCharacter: RequestHandler = async (req, res) => {
  try {
    const characterData = req.body;
    
    const mlResponse = await fetch('http://localhost:5000/classify-character', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(characterData)
    });
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML character classification error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

export const handleMLNBInfo: RequestHandler = async (_req, res) => {
  try {
    const mlResponse = await fetch('http://localhost:5000/nb-info');
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML Naive Bayes info error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

// ===== SVM ENDPOINTS =====

/**
 * POST /api/ml/train-svm
 * Train the SVM classifier
 * 
 * Body:
 *   {
 *     "kernel": "rbf" | "linear" | "poly" | "sigmoid",  // optional, default "rbf"
 *     "optimize": boolean  // optional, default false (uses GridSearchCV if true)
 *   }
 */
export const handleMLTrainSVM: RequestHandler = async (req, res) => {
  try {
    const { kernel, optimize } = req.body;
    
    const mlResponse = await fetch('http://localhost:5000/train-svm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kernel, optimize })
    });
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML SVM training error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

/**
 * POST /api/ml/predict-svm
 * Predict character using SVM
 * 
 * Body:
 *   {
 *     "text": "Character quote or description",
 *     "top_k": 5  // optional, default 5
 *   }
 */
export const handleMLPredictSVM: RequestHandler = async (req, res) => {
  try {
    const { text, top_k } = req.body;
    
    if (!text) {
      return res.status(400).json({
        success: false,
        error: 'Text is required'
      });
    }
    
    const mlResponse = await fetch('http://localhost:5000/predict-svm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, top_k })
    });
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML SVM prediction error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

/**
 * GET /api/ml/svm-feature-importance
 * Get feature importance from SVM (linear kernel only)
 * 
 * Query params:
 *   top_n: number of top features (default 20)
 */
export const handleMLSVMFeatureImportance: RequestHandler = async (req, res) => {
  try {
    const top_n = req.query.top_n || 20;
    
    const mlResponse = await fetch(`http://localhost:5000/svm-feature-importance?top_n=${top_n}`);
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML SVM feature importance error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};

/**
 * GET /api/ml/svm-info
 * Get information about the SVM model
 */
export const handleMLSVMInfo: RequestHandler = async (_req, res) => {
  try {
    const mlResponse = await fetch('http://localhost:5000/svm-info');
    
    if (!mlResponse.ok) {
      throw new Error(`ML service returned ${mlResponse.status}`);
    }
    
    const data = await mlResponse.json();
    res.json(data);
    
  } catch (error) {
    console.error("ML SVM info error:", error);
    res.status(503).json({ 
      success: false,
      error: 'ML service unavailable' 
    });
  }
};
