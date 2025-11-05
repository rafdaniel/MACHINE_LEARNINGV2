import { RequestHandler } from "express";
import { TodayGameResponse } from "@shared/api";
import { getCharacterById } from "../data/characters";
import { getGameState } from "../data/gameState";

/**
 * GET /api/game/today
 * Get today's game state with clues unlocked based on incorrect guesses
 */
export const handleGetTodayGame: RequestHandler = (req, res) => {
  try {
    // In production, get from authenticated user session
    const sessionId = req.headers["x-session-id"] as string || "default-session";
    
    const state = getGameState(sessionId);
    // Use the stored character ID from state (same for all users today)
    const dailyCharacter = getCharacterById(state.characterId);
    
    if (!dailyCharacter) {
      return res.status(500).json({ error: "Character not found" });
    }
    
    const incorrectGuesses = state.guesses.filter((g) => !g.isCorrect).length;
    const attemptsRemaining = -1; // Unlimited attempts
    
    console.log(`[Game State] Session: ${sessionId}`);
    console.log(`[Game State] Total guesses: ${state.guesses.length}`);
    console.log(`[Game State] Incorrect guesses: ${incorrectGuesses}`);
    console.log(`[Game State] Character: ${dailyCharacter.name}`);
    console.log(`[Game State] Visual unlocked: ${state.guesses.length >= 1}`);
    
    // Build clues based on guesses
    // Clue 1 (Picture): Unlocked after 1st guess (correct or incorrect)
    // Clue 2 (Quote): Unlocked after 2nd incorrect guess
    // Clue 3 (Source): Unlocked after 3rd incorrect guess
    const response: TodayGameResponse = {
      date: state.date,
      clues: {
        visual: state.guesses.length >= 1 ? dailyCharacter.imageUrl : null,
        quote: incorrectGuesses >= 2 ? dailyCharacter.quote : null,
        source:
          incorrectGuesses >= 3
            ? {
                title: dailyCharacter.source,
                genre: dailyCharacter.genre,
              }
            : null,
      },
      guesses: state.guesses,
      attemptsRemaining,
      isComplete: state.isComplete,
      isWon: state.isWon,
    };
    
    // If game is complete, reveal the character
    if (state.isComplete) {
      response.revealedCharacter = dailyCharacter;
    }
    
    res.json(response);
  } catch (error) {
    console.error("Error getting today's game:", error);
    res.status(500).json({ error: "Internal server error" });
  }
};

/**
 * POST /api/game/reveal-answer
 * Reveal the correct answer (user gives up)
 */
export const handleRevealAnswer: RequestHandler = (req, res) => {
  try {
    const sessionId = req.headers["x-session-id"] as string || "default-session";
    
    const state = getGameState(sessionId);
    const dailyCharacter = getCharacterById(state.characterId);
    
    if (!dailyCharacter) {
      return res.status(500).json({ error: "Character not found" });
    }
    
    console.log(`[Reveal Answer] Session: ${sessionId} revealed answer: ${dailyCharacter.name}`);
    
    res.json({
      success: true,
      character: dailyCharacter
    });
  } catch (error) {
    console.error("Error revealing answer:", error);
    res.status(500).json({ error: "Internal server error" });
  }
};
