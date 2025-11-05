/**
 * ML Hint System Component
 * Displays AI-powered hints after 4 incorrect guesses
 */

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Lightbulb, Sparkles, Brain, TrendingUp, Eye, AlertTriangle, BookOpen, Cpu, Network, GitBranch, Layers, Zap } from "lucide-react";
import { MLHint } from "@shared/api";

interface MLHintSystemProps {
  incorrectGuesses: number;
  quote: string | null;
  source: { title: string; genre: string } | null;
  correctAnswer: string | null; // The actual character name to reveal
  onHintSelect?: (hint: string) => void;
  onRevealAnswer?: () => void; // Callback when answer is revealed
}

export default function MLHintSystem({ 
  incorrectGuesses, 
  quote, 
  source,
  correctAnswer,
  onHintSelect,
  onRevealAnswer
}: MLHintSystemProps) {
  const [hints, setHints] = useState<MLHint[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRevealDialog, setShowRevealDialog] = useState(false);
  const [answerRevealed, setAnswerRevealed] = useState(false);
  const [showInsights, setShowInsights] = useState(false);

  // Trigger hint fetching when 4th incorrect guess is made
  useEffect(() => {
    if (incorrectGuesses >= 4 && !hasLoaded) {
      fetchHints();
    }
  }, [incorrectGuesses, hasLoaded]);

  const fetchHints = async () => {
    setIsLoading(true);
    setError(null);

    try {
      console.log("🔍 Fetching ML hints with clues:", { quote, source });
      
      const response = await fetch("/api/ml/get-hints", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          character: {
            name: "Unknown",
            quote: quote || "No quote available",
            universe: "Unknown",
            genre: source?.genre || "Unknown",
            powers: [],
            description: `Character from ${source?.title || "unknown source"}`,
            source: source?.title || "Unknown"
          }
        })
      });

      console.log("📡 Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("❌ Response not OK:", errorText);
        throw new Error(`Failed to fetch ML hints (${response.status})`);
      }

      const data = await response.json();
      console.log("📊 ML hints data:", data);
      
      if (data.success && data.hints && data.hints.length > 0) {
        console.log("✅ Received", data.hints.length, "hints");
        setHints(data.hints);
        setHasLoaded(true);
      } else {
        console.warn("⚠️ No hints in response:", data);
        setError(data.error || data.message || "No hints available from AI models. The ML service may need more clues.");
      }
    } catch (err) {
      console.error("❌ ML hint error:", err);
      setError("AI hint system unavailable. Make sure Flask ML service is running on port 5000.");
    } finally {
      setIsLoading(false);
    }
  };

  // Don't show anything until 4 incorrect guesses
  if (incorrectGuesses < 4) {
    return null;
  }

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'knn': return 'bg-green-500/20 text-green-400 border-green-500';
      case 'svm': return 'bg-blue-500/20 text-blue-400 border-blue-500';
      case 'decision_tree': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500';
      case 'ann': return 'bg-red-500/20 text-red-400 border-red-500';
      case 'multiple': return 'bg-purple-500/20 text-purple-400 border-purple-500';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500';
    }
  };

  const getSourceLabel = (source: string) => {
    switch (source) {
      case 'knn': return '🟢 K-NN';
      case 'svm': return '🔵 SVM';
      case 'decision_tree': return '🟡 Tree';
      case 'ann': return '🔴 ANN';
      case 'multiple': return '🟣 Multiple AIs';
      default: return source;
    }
  };

  return (
    <Card className="bg-gradient-to-br from-purple-900/40 to-blue-900/40 border-2 border-purple-500 shadow-xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-white">
              <Brain className="h-6 w-6 text-purple-400 animate-pulse" />
              <Sparkles className="h-5 w-5 text-yellow-400" />
              AI Hint System Activated
            </CardTitle>
            <CardDescription className="text-purple-200">
              You've made {incorrectGuesses} incorrect guesses. Our 6 ML algorithms suggest these characters:
            </CardDescription>
          </div>
          {!isLoading && (
            <Button
              onClick={() => {
                setHasLoaded(false);
                setError(null);
                fetchHints();
              }}
              size="sm"
              variant="outline"
              className="border-purple-500 text-purple-300 hover:bg-purple-900"
            >
              🔄 Refresh
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent>
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="flex flex-col items-center gap-3">
              <Brain className="h-12 w-12 text-purple-400 animate-bounce" />
              <p className="text-purple-300 animate-pulse">
                🤖 Analyzing clues with 6 AI algorithms...
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-500/20 border border-red-500 rounded-lg p-6 text-center space-y-4">
            <div className="flex justify-center">
              <AlertTriangle className="h-12 w-12 text-yellow-400" />
            </div>
            <p className="text-red-300 mb-2">⚠️ {error}</p>
            <p className="text-sm text-gray-400 mb-4">
              Having trouble? You can reveal the answer or keep trying!
            </p>
            <div className="flex gap-3 justify-center">
              <Button
                onClick={fetchHints}
                size="sm"
                variant="outline"
                className="border-purple-500 text-purple-300 hover:bg-purple-900"
              >
                🔄 Try Again
              </Button>
              <Button
                onClick={() => setShowRevealDialog(true)}
                size="sm"
                className="bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700"
              >
                <Eye className="h-4 w-4 mr-2" />
                Reveal Answer
              </Button>
            </div>
          </div>
        )}

        {!isLoading && !error && hints.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm text-purple-300 mb-4">
              <Lightbulb className="h-4 w-4" />
              <span>Click a hint to auto-fill it in your guess:</span>
            </div>

            {hints.map((hint, index) => (
              <button
                key={index}
                onClick={() => onHintSelect?.(hint.character)}
                className="w-full bg-gray-800/50 hover:bg-gray-700/50 border border-purple-500/30 hover:border-purple-500 rounded-lg p-4 transition-all duration-200 hover:scale-105 cursor-pointer group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="bg-purple-500/20 rounded-full w-8 h-8 flex items-center justify-center font-bold text-purple-300">
                      {index + 1}
                    </div>
                    <div className="text-left">
                      <p className="text-white font-semibold text-lg group-hover:text-purple-300 transition-colors">
                        {hint.character}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge 
                          variant="outline" 
                          className={`text-xs ${getSourceColor(hint.source)}`}
                        >
                          {getSourceLabel(hint.source)}
                        </Badge>
                        {hint.source === 'multiple' && (
                          <span className="text-xs text-purple-400">
                            ⭐ Multiple AI models agree!
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="text-right">
                      <p className="text-sm text-gray-400">Confidence</p>
                      <div className="flex items-center gap-1">
                        <TrendingUp className="h-4 w-4 text-green-400" />
                        <p className="text-xl font-bold text-green-400">
                          {(hint.confidence * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </button>
            ))}

            <div className="mt-4 text-center text-xs text-purple-300 bg-purple-900/30 rounded-lg p-3">
              💡 These hints are generated by analyzing your clues with K-NN, SVM, Decision Tree, and Neural Network algorithms
            </div>
          </div>
        )}

        {!isLoading && !error && hints.length === 0 && hasLoaded && (
          <div className="text-center py-8 space-y-4">
            <div className="flex justify-center">
              <Brain className="h-16 w-16 text-gray-500 opacity-50" />
            </div>
            <p className="text-gray-400 mb-2">Not enough clues available yet for AI hints.</p>
            <p className="text-sm text-gray-500 mb-4">The ML models need more information to make accurate predictions.</p>
            <div className="flex gap-3 justify-center">
              <Button
                onClick={() => {
                  setHasLoaded(false);
                  fetchHints();
                }}
                size="sm"
                variant="outline"
                className="border-purple-500 text-purple-300 hover:bg-purple-900"
              >
                🔄 Try Again
              </Button>
              <Button
                onClick={() => setShowRevealDialog(true)}
                size="sm"
                className="bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700"
              >
                <Eye className="h-4 w-4 mr-2" />
                Reveal Answer
              </Button>
            </div>
          </div>
        )}

        {answerRevealed && correctAnswer && (
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border-2 border-yellow-500 rounded-lg p-6 text-center space-y-3">
              <div className="flex justify-center">
                <Sparkles className="h-12 w-12 text-yellow-400 animate-pulse" />
              </div>
              <h3 className="text-2xl font-bold text-yellow-300">Answer Revealed!</h3>
              <p className="text-3xl font-bold text-white tracking-wide">{correctAnswer}</p>
              <p className="text-sm text-gray-300 mt-2">
                Better luck next time! Start a new game to try again.
              </p>
              <Button
                onClick={() => setShowInsights(!showInsights)}
                className="mt-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              >
                <BookOpen className="h-4 w-4 mr-2" />
                {showInsights ? "Hide AI Insights" : "Show AI Insights"}
              </Button>
            </div>

            {/* AI Insights Section */}
            {showInsights && (
              <div className="bg-gradient-to-br from-blue-900/40 to-purple-900/40 border-2 border-blue-500 rounded-lg p-6 space-y-6 animate-in fade-in duration-500">
                <div className="text-center">
                  <h3 className="text-2xl font-bold text-blue-300 flex items-center justify-center gap-2">
                    <Brain className="h-6 w-6" />
                    How Our AI Analyzed This Challenge
                  </h3>
                  <p className="text-sm text-gray-300 mt-2">
                    Learn how 6 different machine learning algorithms worked together to solve this puzzle
                  </p>
                </div>

                {/* Algorithm Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* K-NN */}
                  <div className="bg-gray-800/50 border border-green-500/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="bg-green-500/20 p-2 rounded">
                        <Network className="h-5 w-5 text-green-400" />
                      </div>
                      <h4 className="font-bold text-green-400">K-Nearest Neighbors (K-NN)</h4>
                    </div>
                    <p className="text-sm text-gray-300">
                      <span className="font-semibold text-white">How it works:</span> Finds the most similar characters based on quote patterns, genre, and source material.
                    </p>
                    <p className="text-xs text-gray-400">
                      💡 Think of it as "Which characters are most like this one based on their traits?"
                    </p>
                    <div className="bg-green-500/10 rounded p-2 text-xs text-green-300">
                      <strong>Real-world use:</strong> Recommendation systems (Netflix, Spotify), Image recognition
                    </div>
                  </div>

                  {/* SVM */}
                  <div className="bg-gray-800/50 border border-blue-500/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="bg-blue-500/20 p-2 rounded">
                        <GitBranch className="h-5 w-5 text-blue-400" />
                      </div>
                      <h4 className="font-bold text-blue-400">Support Vector Machine (SVM)</h4>
                    </div>
                    <p className="text-sm text-gray-300">
                      <span className="font-semibold text-white">How it works:</span> Creates decision boundaries to separate characters into categories with high confidence.
                    </p>
                    <p className="text-xs text-gray-400">
                      💡 Excels at binary classification like "Hero vs Villain" or "Marvel vs DC"
                    </p>
                    <div className="bg-blue-500/10 rounded p-2 text-xs text-blue-300">
                      <strong>Real-world use:</strong> Text classification, Face detection, Medical diagnosis
                    </div>
                  </div>

                  {/* Decision Tree */}
                  <div className="bg-gray-800/50 border border-yellow-500/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="bg-yellow-500/20 p-2 rounded">
                        <Layers className="h-5 w-5 text-yellow-400" />
                      </div>
                      <h4 className="font-bold text-yellow-400">Decision Tree</h4>
                    </div>
                    <p className="text-sm text-gray-300">
                      <span className="font-semibold text-white">How it works:</span> Makes yes/no decisions like "Is it from Marvel?" → "Has super strength?" → Character!
                    </p>
                    <p className="text-xs text-gray-400">
                      💡 Most interpretable algorithm - you can literally see the decision path
                    </p>
                    <div className="bg-yellow-500/10 rounded p-2 text-xs text-yellow-300">
                      <strong>Real-world use:</strong> Credit scoring, Customer segmentation, Game AI
                    </div>
                  </div>

                  {/* ANN */}
                  <div className="bg-gray-800/50 border border-red-500/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="bg-red-500/20 p-2 rounded">
                        <Cpu className="h-5 w-5 text-red-400" />
                      </div>
                      <h4 className="font-bold text-red-400">Artificial Neural Network (ANN)</h4>
                    </div>
                    <p className="text-sm text-gray-300">
                      <span className="font-semibold text-white">How it works:</span> Learns complex patterns through layers of "neurons" that activate based on input features.
                    </p>
                    <p className="text-xs text-gray-400">
                      💡 Inspired by the human brain - can learn non-linear relationships
                    </p>
                    <div className="bg-red-500/10 rounded p-2 text-xs text-red-300">
                      <strong>Real-world use:</strong> ChatGPT, Image generation, Self-driving cars
                    </div>
                  </div>

                  {/* Naive Bayes */}
                  <div className="bg-gray-800/50 border border-purple-500/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="bg-purple-500/20 p-2 rounded">
                        <Zap className="h-5 w-5 text-purple-400" />
                      </div>
                      <h4 className="font-bold text-purple-400">Naive Bayes</h4>
                    </div>
                    <p className="text-sm text-gray-300">
                      <span className="font-semibold text-white">How it works:</span> Uses probability to predict categories - calculates likelihood based on word patterns.
                    </p>
                    <p className="text-xs text-gray-400">
                      💡 Fast and effective for text classification with limited data
                    </p>
                    <div className="bg-purple-500/10 rounded p-2 text-xs text-purple-300">
                      <strong>Real-world use:</strong> Spam filters, Sentiment analysis, Document categorization
                    </div>
                  </div>

                  {/* Linear Regression */}
                  <div className="bg-gray-800/50 border border-cyan-500/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="bg-cyan-500/20 p-2 rounded">
                        <TrendingUp className="h-5 w-5 text-cyan-400" />
                      </div>
                      <h4 className="font-bold text-cyan-400">Linear Regression</h4>
                    </div>
                    <p className="text-sm text-gray-300">
                      <span className="font-semibold text-white">How it works:</span> Predicts difficulty scores by finding relationships between clue complexity and guess attempts.
                    </p>
                    <p className="text-xs text-gray-400">
                      💡 Helps balance game difficulty and analyze which characters are hardest to guess
                    </p>
                    <div className="bg-cyan-500/10 rounded p-2 text-xs text-cyan-300">
                      <strong>Real-world use:</strong> Stock price prediction, Sales forecasting, Risk assessment
                    </div>
                  </div>
                </div>

                {/* Ensemble Learning Explanation */}
                <div className="bg-gradient-to-r from-purple-800/30 to-blue-800/30 border border-purple-500/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-300 flex items-center gap-2 mb-2">
                    <Sparkles className="h-5 w-5" />
                    Ensemble Learning: Combining Multiple AI Models
                  </h4>
                  <p className="text-sm text-gray-300 mb-2">
                    Our hint system uses <strong>ensemble learning</strong> - combining predictions from multiple algorithms to get more accurate results!
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                    <div className="bg-gray-800/50 rounded p-2">
                      <strong className="text-white">🎯 Higher Accuracy:</strong>
                      <p className="text-gray-400">Multiple models catch errors others miss</p>
                    </div>
                    <div className="bg-gray-800/50 rounded p-2">
                      <strong className="text-white">🛡️ More Robust:</strong>
                      <p className="text-gray-400">Less affected by noisy or incomplete data</p>
                    </div>
                    <div className="bg-gray-800/50 rounded p-2">
                      <strong className="text-white">🎓 Better Learning:</strong>
                      <p className="text-gray-400">Each algorithm specializes in different patterns</p>
                    </div>
                  </div>
                </div>

                {/* What You Learned */}
                <div className="bg-gradient-to-r from-green-800/30 to-teal-800/30 border border-green-500/50 rounded-lg p-4">
                  <h4 className="font-bold text-green-300 flex items-center gap-2 mb-2">
                    <BookOpen className="h-5 w-5" />
                    What You Just Experienced
                  </h4>
                  <ul className="text-sm text-gray-300 space-y-2 ml-4">
                    <li className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">✓</span>
                      <span><strong>Feature Engineering:</strong> The system converted clues (quotes, sources) into numerical features AI can understand</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">✓</span>
                      <span><strong>Text Vectorization:</strong> Quotes were transformed into mathematical vectors using TF-IDF and embeddings</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">✓</span>
                      <span><strong>Confidence Scoring:</strong> Each algorithm provided a confidence percentage showing how certain it was</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">✓</span>
                      <span><strong>Voting Mechanism:</strong> When multiple AIs agreed, that hint got ranked higher (wisdom of crowds!)</span>
                    </li>
                  </ul>
                </div>

                {/* Fun Facts */}
                <div className="bg-gray-800/30 border border-gray-600 rounded-lg p-4">
                  <h4 className="font-bold text-gray-300 mb-2">🎮 Fun Facts About This Game's AI</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-gray-400">
                    <div>• Trained on {hints.length > 0 ? "50+" : "multiple"} superhero characters</div>
                    <div>• Processes quotes in milliseconds using NLP</div>
                    <div>• Uses scikit-learn & sentence transformers</div>
                    <div>• Updates predictions as you unlock more clues</div>
                    <div>• Combines 6 algorithms for maximum accuracy</div>
                    <div>• Real production-level ML architecture</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>

      {/* Reveal Answer Confirmation Dialog */}
      <AlertDialog open={showRevealDialog} onOpenChange={setShowRevealDialog}>
        <AlertDialogContent className="bg-gray-900 border-2 border-yellow-500">
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2 text-yellow-300 text-xl">
              <Eye className="h-6 w-6" />
              Reveal the Answer?
            </AlertDialogTitle>
            <AlertDialogDescription className="text-gray-300 text-base space-y-3">
              <p>
                Are you sure you want to reveal the correct answer? 
              </p>
              <p className="text-yellow-400 font-semibold">
                ⚠️ This will end your current game and show you the character.
              </p>
              <p className="text-sm text-gray-400">
                You can always start a new game with a different character!
              </p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-gray-700 hover:bg-gray-600 border-gray-600">
              Keep Guessing
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                setAnswerRevealed(true);
                setShowRevealDialog(false);
                onRevealAnswer?.();
              }}
              className="bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700"
            >
              <Eye className="h-4 w-4 mr-2" />
              Yes, Reveal Answer
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  );
}
