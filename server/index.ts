import "dotenv/config";
import express from "express";
import cors from "cors";
import { handleDemo } from "./routes/demo";
import { handleGetTodayGame } from "./routes/game";
import { handleGuess } from "./routes/guess";
import { handleGetCharacters } from "./routes/characters";
import { 
  handleMLPredict, 
  handleMLAnalyzeGame, 
  handleMLHealth,
  handleMLPredictDifficulty,
  handleMLDifficultyRankings,
  handleMLFeatureImportance,
  handleMLPredictGenre,
  handleMLPredictUniverse,
  handleMLClassifyCharacter,
  handleMLNBInfo,
  handleMLTrainSVM,
  handleMLPredictSVM,
  handleMLSVMFeatureImportance,
  handleMLSVMInfo
} from "./routes/ml";

export function createServer() {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Example API routes
  app.get("/api/ping", (_req, res) => {
    const ping = process.env.PING_MESSAGE ?? "ping";
    res.json({ message: ping });
  });

  app.get("/api/demo", handleDemo);

  // Game API routes
  app.get("/api/game/today", handleGetTodayGame);
  app.post("/api/game/guess", handleGuess);
  app.get("/api/characters", handleGetCharacters);

  // ML API routes - K-NN
  app.post("/api/ml/predict", handleMLPredict);
  app.post("/api/ml/analyze-game", handleMLAnalyzeGame);
  app.get("/api/ml/health", handleMLHealth);
  
  // ML API routes - Linear Regression
  app.post("/api/ml/predict-difficulty", handleMLPredictDifficulty);
  app.get("/api/ml/difficulty-rankings", handleMLDifficultyRankings);
  app.get("/api/ml/feature-importance", handleMLFeatureImportance);
  
  // ML API routes - Naive Bayes
  app.post("/api/ml/predict-genre", handleMLPredictGenre);
  app.post("/api/ml/predict-universe", handleMLPredictUniverse);
  app.post("/api/ml/classify-character", handleMLClassifyCharacter);
  app.get("/api/ml/nb-info", handleMLNBInfo);
  
  // ML API routes - SVM
  app.post("/api/ml/train-svm", handleMLTrainSVM);
  app.post("/api/ml/predict-svm", handleMLPredictSVM);
  app.get("/api/ml/svm-feature-importance", handleMLSVMFeatureImportance);
  app.get("/api/ml/svm-info", handleMLSVMInfo);

  return app;
}
