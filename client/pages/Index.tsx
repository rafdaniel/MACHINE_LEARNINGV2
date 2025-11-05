import { useState, useEffect, useRef } from "react";
import ParticlesBackground from "@/components/ParticlesBackground";
import GameHeader from "@/components/GameHeader";
import CluePanel from "@/components/CluePanel";
import AutocompleteInput from "@/components/AutocompleteInput";
import MLHintSystem from "@/components/MLHintSystem";
import CharacterSimilarityExplorer from "@/components/CharacterSimilarityExplorer";
import { TodayGameResponse, GuessResponse } from "@shared/api";
import { useToast } from "@/hooks/use-toast";

export default function Index() {
  const [gameState, setGameState] = useState<TodayGameResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [hintCharacter, setHintCharacter] = useState<string>("");
  const [correctAnswer, setCorrectAnswer] = useState<string | null>(null);
  const [revealedCharacterId, setRevealedCharacterId] = useState<string | null>(null);
  const { toast } = useToast();
  const inputRef = useRef<any>(null);

  // Generate or get session ID
  const getSessionId = () => {
    let sessionId = localStorage.getItem("session-id");
    if (!sessionId) {
      sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem("session-id", sessionId);
    }
    return sessionId;
  };

  // Fetch today's game state
  const fetchGameState = async () => {
    try {
      const sessionId = getSessionId();
      const response = await fetch("/api/game/today", {
        headers: {
          "X-Session-Id": sessionId,
        },
      });
      
      if (!response.ok) {
        throw new Error("Failed to fetch game state");
      }
      
      const data: TodayGameResponse = await response.json();
      setGameState(data);
    } catch (error) {
      console.error("Error fetching game state:", error);
      toast({
        title: "Error",
        description: "Failed to load game. Please refresh the page.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchGameState();
  }, []);

  // Start a new game with a random character
  const handleNewGame = () => {
    // Generate a new session ID to get a new random character
    const newSessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem("session-id", newSessionId);
    
    // Refresh the game state
    setIsLoading(true);
    fetchGameState();
    
    toast({
      title: "New Game Started!",
      description: "A new random character has been selected. Good luck!",
      variant: "default",
    });
  };

  // Handle hint selection from ML system
  const handleHintSelect = (character: string) => {
    setHintCharacter(character);
    toast({
      title: "💡 Hint Selected",
      description: `"${character}" has been filled in. Click Submit to guess!`,
      variant: "default",
    });
    // Scroll to input
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  // Handle revealing the answer
  const handleRevealAnswer = async () => {
    try {
      const sessionId = getSessionId();
      const response = await fetch("/api/game/reveal-answer", {
        method: "POST",
        headers: {
          "X-Session-Id": sessionId,
        },
      });

      if (!response.ok) {
        throw new Error("Failed to reveal answer");
      }

      const data = await response.json();
      if (data.success && data.character) {
        setCorrectAnswer(data.character.name);
        setRevealedCharacterId(data.character.id); // Store character ID for similarity explorer
      }

      toast({
        title: "Answer Revealed",
        description: `The character is ${data.character.name}!`,
        variant: "default",
      });
    } catch (error) {
      console.error("Error revealing answer:", error);
      toast({
        title: "Error",
        description: "Failed to reveal answer",
        variant: "destructive",
      });
    }
  };

  const handleGuessSubmit = async (guess: string) => {
    if (!gameState || gameState.isComplete || isSubmitting) return;
    
    setIsSubmitting(true);
    
    try {
      const sessionId = getSessionId();
      const response = await fetch("/api/game/guess", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-Id": sessionId,
        },
        body: JSON.stringify({ guess }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to submit guess");
      }
      
      const data: GuessResponse = await response.json();
      
      // Clear hint after submission
      setHintCharacter("");
      
      // Refresh game state to get updated clues
      await fetchGameState();
      
      // Show feedback
      if (data.isCorrect) {
        toast({
          title: "🎉 Correct!",
          description: `You guessed it! The character was ${data.revealedCharacter?.name}!`,
          variant: "default",
        });
      } else if (data.isGameOver) {
        toast({
          title: "Game Over",
          description: `The character was ${data.revealedCharacter?.name}. Try again tomorrow!`,
          variant: "destructive",
        });
      } else {
        toast({
          title: "Incorrect",
          description: `${data.attemptsRemaining} attempts remaining`,
          variant: "default",
        });
      }
    } catch (error) {
      console.error("Error submitting guess:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to submit guess",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-netflix-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-netflix-red mx-auto mb-4"></div>
          <p className="text-gray-400">Loading game...</p>
        </div>
      </div>
    );
  }

  if (!gameState) {
    return (
      <div className="min-h-screen bg-netflix-black text-white flex items-center justify-center">
        <div className="text-center">
          <p className="text-netflix-red font-bold text-xl mb-2">Error</p>
          <p className="text-gray-400">Failed to load game</p>
        </div>
      </div>
    );
  }

  const incorrectGuesses = gameState.guesses.filter((g) => !g.isCorrect).length;

  return (
    <div className="min-h-screen bg-netflix-black text-white relative overflow-hidden">
      <ParticlesBackground />

      <div className="content-wrapper min-h-screen flex flex-col">
        <GameHeader />

        <main className="flex-1 overflow-auto px-4 sm:px-8 py-8">
          <div className="max-w-6xl mx-auto">
            {/* Clue Panel */}
            <CluePanel 
              incorrectGuesses={incorrectGuesses}
              clues={gameState.clues}
            />

            {/* ML Hint System - Appears after 4 incorrect guesses */}
            {!gameState.isComplete && (
              <div className="mb-8">
                <MLHintSystem
                  incorrectGuesses={incorrectGuesses}
                  quote={gameState.clues.quote}
                  source={gameState.clues.source}
                  correctAnswer={correctAnswer}
                  onHintSelect={handleHintSelect}
                  onRevealAnswer={handleRevealAnswer}
                />
              </div>
            )}

            {/* Game Status */}
            <div className="mb-8 text-center">
              <p className="text-gray-400 text-sm mb-4">
                {!gameState.isComplete ? (
                  <>
                    <span className="text-netflix-red font-bold">
                      {gameState.guesses.length}
                    </span>
                    {" guesses made • Unlimited attempts"}
                  </>
                ) : gameState.isWon ? (
                  <span className="text-green-500 font-bold text-lg">
                    🎉 You Won! The character was {gameState.revealedCharacter?.name} from {gameState.revealedCharacter?.universe} • {gameState.guesses.length} guesses
                  </span>
                ) : (
                  <span className="text-netflix-red font-bold">
                    Game Over - The character was {gameState.revealedCharacter?.name} from {gameState.revealedCharacter?.universe}
                  </span>
                )}
              </p>
              
              {/* New Game Button */}
              <button
                onClick={handleNewGame}
                className="px-6 py-2 bg-netflix-red hover:bg-red-700 text-white font-bold rounded-lg transition-colors duration-200 shadow-lg hover:shadow-xl"
              >
                🎮 New Random Game
              </button>
            </div>

            {/* Character Display - Only show when game is won */}
            {gameState.isComplete && gameState.isWon && gameState.revealedCharacter && (
              <div className="relative z-20 mb-8 flex justify-center">
                <div className="max-w-md w-full bg-gradient-to-br from-primary/20 to-transparent rounded-lg p-6 border-2 border-primary backdrop-blur-sm">
                  <div className="text-center">
                    <h2 className="text-primary text-lg font-bold uppercase tracking-wider mb-4">
                      🎉 Character Revealed!
                    </h2>
                    <div className="aspect-square overflow-hidden rounded-lg mb-4 border-2 border-primary/50">
                      <img 
                        src={gameState.revealedCharacter.characterImageUrl || gameState.revealedCharacter.imageUrl} 
                        alt={gameState.revealedCharacter.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <h3 className="text-white text-2xl font-bold mb-2">
                      {gameState.revealedCharacter.name}
                    </h3>
                    
                    {/* Universe Badge - Prominent Display */}
                    <div className="flex items-center justify-center gap-2 mb-3">
                      <span 
                        className={`px-4 py-2 rounded-full text-sm font-bold uppercase tracking-wider border-2 ${
                          gameState.revealedCharacter.universe === "Marvel"
                            ? "bg-red-500/20 text-red-400 border-red-500"
                            : gameState.revealedCharacter.universe === "DC"
                            ? "bg-blue-500/20 text-blue-400 border-blue-500"
                            : "bg-purple-500/20 text-purple-400 border-purple-500"
                        }`}
                      >
                        {gameState.revealedCharacter.universe} Universe
                      </span>
                    </div>
                    
                    <p className="text-gray-400 text-sm mb-2">
                      {gameState.revealedCharacter.attributes.alignment} • {gameState.revealedCharacter.genre}
                    </p>
                    
                    {/* Powers Section */}
                    {gameState.revealedCharacter.attributes.powers && gameState.revealedCharacter.attributes.powers.length > 0 && (
                      <div className="bg-primary/10 rounded px-4 py-3 mt-4 mb-4">
                        <h4 className="text-primary text-xs uppercase font-bold tracking-wider mb-2">
                          ⚡ Powers & Abilities
                        </h4>
                        <div className="flex flex-wrap gap-2 justify-center">
                          {gameState.revealedCharacter.attributes.powers.map((power, index) => (
                            <span
                              key={index}
                              className="px-3 py-1 bg-primary/20 text-white text-xs rounded-full border border-primary/30"
                            >
                              {power}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    <div className="bg-primary/10 rounded px-4 py-3 mt-4">
                      <p className="text-white/90 italic text-sm">
                        "{gameState.revealedCharacter.quote}"
                      </p>
                      <p className="text-gray-500 text-xs mt-2">
                        — {gameState.revealedCharacter.source}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Character Similarity Explorer - Shows after winning OR when answer is revealed */}
            {((gameState.isComplete && gameState.isWon && gameState.revealedCharacter) || (correctAnswer && revealedCharacterId)) && (
              <div className="relative z-20 mb-8">
                <CharacterSimilarityExplorer
                  characterId={gameState.revealedCharacter?.id || revealedCharacterId || ""}
                  characterName={gameState.revealedCharacter?.name || correctAnswer || ""}
                  onPlayCharacter={(charId) => {
                    console.log("Play character:", charId);
                    toast({
                      title: "Feature Coming Soon",
                      description: "Direct character selection will be available in the next update!",
                      variant: "default",
                    });
                  }}
                />
              </div>
            )}

            {/* Input Section */}
            <div className="mb-8">
              <AutocompleteInput
                onSubmit={handleGuessSubmit}
                disabled={gameState.isComplete || isSubmitting}
                defaultValue={hintCharacter}
              />
            </div>

            {/* Previous Guesses History */}
            {gameState.guesses.length > 0 && (
              <div className="relative z-20 mb-8">
                <h3 className="text-primary text-sm uppercase font-bold tracking-wider mb-3 text-center">
                  Previous Guesses ({gameState.guesses.length})
                </h3>
                <div className="max-w-2xl mx-auto bg-background/30 backdrop-blur-sm rounded-lg p-4 border border-primary/20">
                  <div className="flex flex-wrap gap-2 justify-center">
                    {gameState.guesses.map((guess, index) => (
                      <div
                        key={index}
                        className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                          guess.isCorrect
                            ? "bg-green-500/20 text-green-400 border-2 border-green-500"
                            : "bg-red-500/10 text-gray-300 border border-red-500/30"
                        }`}
                      >
                        {guess.isCorrect && "✓ "}
                        {guess.guess}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Game Instructions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-400 mt-12 pt-8 border-t border-gray-800">
              <div className="text-center md:text-left">
                <p className="text-netflix-red font-bold mb-2">How to Play</p>
                <p>Guess the Marvel/DC character with help from unlockable clues.</p>
              </div>
              <div className="text-center">
                <p className="text-netflix-red font-bold mb-2">Unlimited Attempts</p>
                <p>Keep guessing until you find the right character/superheroes!</p>
              </div>
              <div className="text-center md:text-right">
                <p className="text-netflix-red font-bold mb-2">Unlock Clues</p>
                <p>Picture (1st guess) • Quote (2nd wrong) • Source (3rd wrong)</p>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
