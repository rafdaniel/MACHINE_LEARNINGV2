/**
 * Shared code between client and server
 * Useful to share types between client and server
 * and/or small pure JS functions that can be used on both client and server
 */

/**
 * Example response type for /api/demo
 */
export interface DemoResponse {
  message: string;
}

/**
 * Character interface for the guessing game
 */
export interface Character {
  id: string;
  name: string;
  aliases: string[]; // Alternate names that should also match
  universe: 'Marvel' | 'DC' | 'Other';
  quote: string;
  source: string; // e.g., "The Avengers", "Batman: The Dark Knight"
  genre: string; // e.g., "Superhero", "Action", "Fantasy"
  imageUrl: string; // Clue image (symbolic representation)
  characterImageUrl: string; // Actual character image (revealed on win)
  pixelatedImageUrl?: string; // Pre-pixelated version or will be generated
  attributes: {
    alignment: 'hero' | 'villain' | 'anti-hero';
    powers: string[];
    team?: string;
    firstAppearance?: string;
  };
}

/**
 * Daily challenge interface
 */
export interface DailyChallenge {
  date: string; // YYYY-MM-DD format
  character: Character;
}

/**
 * Game state for a player
 */
export interface GameState {
  date: string;
  characterId: string;
  guesses: GuessResult[];
  isComplete: boolean;
  isWon: boolean;
}

/**
 * Result of a single guess
 */
export interface GuessResult {
  guess: string;
  isCorrect: boolean;
  timestamp: string;
}

/**
 * Response when submitting a guess
 */
export interface GuessResponse {
  isCorrect: boolean;
  isGameOver: boolean;
  isWon: boolean;
  attemptsRemaining: number;
  guessResult: GuessResult;
  revealedCharacter?: Character; // Only sent if game is over
}

/**
 * Response for getting today's game state
 */
export interface TodayGameResponse {
  date: string;
  clues: {
    visual: string | null; // Image URL or null if locked
    quote: string | null; // Quote or null if locked
    source: {
      title: string;
      genre: string;
    } | null; // Source info or null if locked
  };
  guesses: GuessResult[];
  attemptsRemaining: number;
  isComplete: boolean;
  isWon: boolean;
  revealedCharacter?: Character; // Only if game is complete
}

/**
 * Character list response for autocomplete
 */
export interface CharacterListResponse {
  characters: Array<{
    id: string;
    name: string;
    aliases: string[];
  }>;
}

/**
 * ML Hint System interfaces
 */
export interface MLHint {
  character: string;
  confidence: number;
  source: 'knn' | 'svm' | 'decision_tree' | 'ann' | 'multiple';
}

export interface MLHintResponse {
  success: boolean;
  hints: MLHint[];
  message: string;
  error?: string;
}
