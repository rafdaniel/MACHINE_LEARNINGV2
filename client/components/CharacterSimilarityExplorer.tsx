/**
 * Character Similarity Explorer Component
 * Uses K-NN algorithm to show similar characters after winning
 */

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Network, Sparkles, TrendingUp, Users, Zap } from "lucide-react";

interface SimilarCharacter {
  id: string;
  name: string;
  similarity: number;
  distance: number;
  shared_attributes: string[];
  character: {
    id: string;
    name: string;
    universe: string;
    genre: string;
    quote: string;
    source: string;
    imageUrl?: string;
    characterImageUrl?: string;
    attributes?: {
      alignment: string;
      powers: string[];
      team?: string;
    };
  };
}

interface CharacterSimilarityExplorerProps {
  characterId: string;
  characterName: string;
  onPlayCharacter?: (characterId: string) => void;
}

export default function CharacterSimilarityExplorer({
  characterId,
  characterName,
  onPlayCharacter
}: CharacterSimilarityExplorerProps) {
  const [similarCharacters, setSimilarCharacters] = useState<SimilarCharacter[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);

  const fetchSimilarCharacters = async () => {
    setIsLoading(true);
    setError(null);

    try {
      console.log("🔍 Fetching similar characters for:", characterId);

      const response = await fetch("/api/ml/find-similar-characters", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          characterId: characterId,
          top_k: 5
        })
      });

      console.log("📡 Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("❌ Response not OK:", errorText);
        throw new Error(`Failed to fetch similar characters (${response.status})`);
      }

      const data = await response.json();
      console.log("📊 Similar characters data:", data);

      if (data.success && data.similarCharacters) {
        console.log("✅ Received", data.similarCharacters.length, "similar characters");
        setSimilarCharacters(data.similarCharacters);
      } else {
        setError(data.error || "No similar characters found");
      }
    } catch (err) {
      console.error("❌ Similar characters error:", err);
      setError("Failed to load similar characters. Make sure the ML service is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 80) return "text-green-400 bg-green-500/20 border-green-500";
    if (similarity >= 60) return "text-blue-400 bg-blue-500/20 border-blue-500";
    if (similarity >= 40) return "text-yellow-400 bg-yellow-500/20 border-yellow-500";
    return "text-gray-400 bg-gray-500/20 border-gray-500";
  };

  const getSimilarityLabel = (similarity: number) => {
    if (similarity >= 80) return "Very Similar";
    if (similarity >= 60) return "Similar";
    if (similarity >= 40) return "Somewhat Similar";
    return "Less Similar";
  };

  if (!isVisible) {
    return (
      <div className="text-center py-4">
        <Button
          onClick={() => {
            setIsVisible(true);
            fetchSimilarCharacters();
          }}
          className="bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700"
          size="lg"
        >
          <Network className="h-5 w-5 mr-2" />
          Explore Similar Characters
        </Button>
        <p className="text-xs text-gray-400 mt-2">
          Uses K-Nearest Neighbors to find characters with similar attributes
        </p>
      </div>
    );
  }

  return (
    <Card className="bg-gradient-to-br from-green-900/40 to-teal-900/40 border-2 border-green-500 shadow-xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-white">
              <Network className="h-6 w-6 text-green-400 animate-pulse" />
              <Sparkles className="h-5 w-5 text-teal-400" />
              Character Similarity Explorer
            </CardTitle>
            <CardDescription className="text-green-200">
              AI Assistance uses these characters similar to <strong>{characterName}</strong>
            </CardDescription>
          </div>
          <Button
            onClick={() => setIsVisible(false)}
            size="sm"
            variant="outline"
            className="border-green-500 text-green-300 hover:bg-green-900"
          >
            Hide
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="flex flex-col items-center gap-3">
              <Network className="h-12 w-12 text-green-400 animate-bounce" />
              <p className="text-green-300 animate-pulse">
                🤖 Analyzing character relationships with K-NN...
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 text-center">
            <p className="text-red-300 mb-2">⚠️ {error}</p>
            <Button
              onClick={fetchSimilarCharacters}
              size="sm"
              className="bg-red-600 hover:bg-red-700"
            >
              Try Again
            </Button>
          </div>
        )}

        {!isLoading && !error && similarCharacters.length > 0 && (
          <div className="space-y-4">
            {/* Algorithm Info Banner */}
            <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-3 text-sm">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="h-4 w-4 text-green-400" />
                <strong className="text-green-300">Machine Learning Algorithm</strong>
              </div>
              <p className="text-gray-300 text-xs">
                Finds characters with the most similar features based on universe, attributes, powers, and text embeddings.
                Similarity is calculated using cosine distance in high-dimensional feature space.
              </p>
            </div>

            {/* Similar Characters Grid */}
            <div className="grid grid-cols-1 gap-4">
              {similarCharacters.map((similar, index) => (
                <div
                  key={similar.id}
                  className="bg-gray-800/50 border border-green-500/30 rounded-lg p-4 hover:border-green-500 transition-all duration-200 hover:scale-[1.02]"
                >
                  <div className="flex items-start justify-between gap-4">
                    {/* Character Info */}
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <div className="bg-green-500/20 rounded-full w-8 h-8 flex items-center justify-center font-bold text-green-300">
                          {index + 1}
                        </div>
                        <div>
                          <h3 className="text-lg font-bold text-white">
                            {similar.name}
                          </h3>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge variant="outline" className="text-xs bg-gray-700/50">
                              {similar.character.universe}
                            </Badge>
                            <Badge variant="outline" className="text-xs bg-gray-700/50">
                              {similar.character.genre}
                            </Badge>
                          </div>
                        </div>
                      </div>

                      {/* Quote */}
                      <p className="text-sm text-gray-300 italic mb-2">
                        "{similar.character.quote}"
                      </p>

                      {/* Shared Attributes */}
                      <div className="space-y-1 mb-3">
                        <p className="text-xs font-semibold text-green-400">
                          Why they're similar:
                        </p>
                        {similar.shared_attributes && similar.shared_attributes.length > 0 ? (
                          <ul className="space-y-1">
                            {similar.shared_attributes.map((attr, idx) => (
                              <li key={idx} className="text-xs text-gray-400 flex items-start gap-1">
                                <span className="text-green-400">✓</span>
                                <span>{attr}</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-xs text-gray-400">Similar characteristics detected</p>
                        )}
                      </div>

                      {/* Powers */}
                      {similar.character.attributes?.powers && similar.character.attributes.powers.length > 0 && (
                        <div className="flex flex-wrap gap-1 mb-2">
                          {similar.character.attributes.powers.slice(0, 4).map((power, idx) => (
                            <span
                              key={idx}
                              className="text-xs px-2 py-1 bg-teal-500/20 text-teal-300 rounded"
                            >
                              {power}
                            </span>
                          ))}
                          {similar.character.attributes.powers.length > 4 && (
                            <span className="text-xs px-2 py-1 bg-gray-700 text-gray-400 rounded">
                              +{similar.character.attributes.powers.length - 4} more
                            </span>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Similarity Score */}
                    <div className="flex flex-col items-end gap-2">
                      <div className={`px-3 py-1 rounded-full border ${getSimilarityColor(similar.similarity)}`}>
                        <div className="flex items-center gap-1">
                          <TrendingUp className="h-3 w-3" />
                          <span className="text-sm font-bold">{similar.similarity}%</span>
                        </div>
                      </div>
                      <span className="text-xs text-gray-400">
                        {getSimilarityLabel(similar.similarity)}
                      </span>

                      {onPlayCharacter && (
                        <Button
                          onClick={() => onPlayCharacter(similar.id)}
                          size="sm"
                          className="mt-2 bg-green-600 hover:bg-green-700"
                        >
                          Play This Character
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Educational Footer */}
            <div className="bg-gradient-to-r from-teal-800/30 to-green-800/30 border border-teal-500/50 rounded-lg p-4 mt-4">
              <h4 className="font-bold text-teal-300 flex items-center gap-2 mb-2">
                <Users className="h-4 w-4" />
                How K-NN Finds Similar Characters
              </h4>
              <div className="text-xs text-gray-300 space-y-2">
                <p>
                  <strong>1. Feature Extraction:</strong> Converts character attributes (quotes, universe, powers) into numerical vectors using sentence transformers.
                </p>
                <p>
                  <strong>2. Distance Calculation:</strong> Measures cosine distance between vectors in high-dimensional space.
                </p>
                <p>
                  <strong>3. Neighbor Selection:</strong> Returns the K characters with the smallest distances (most similar features).
                </p>
                <p className="text-teal-300">
                  💡 The closer to 100%, the more similar the characters are in their attributes and characteristics!
                </p>
              </div>
            </div>
          </div>
        )}

        {!isLoading && !error && similarCharacters.length === 0 && (
          <div className="text-center py-8">
            <Network className="h-16 w-16 text-gray-500 opacity-50 mx-auto mb-4" />
            <p className="text-gray-400 mb-2">No similar characters found.</p>
            <p className="text-sm text-gray-500">Try analyzing a different character!</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
