# Reveal Answer Feature

## Overview
Added a "Reveal Answer" button that appears when the ML hint system doesn't have enough clues to provide predictions. Users can click this button to reveal the correct character after confirming they want to give up.

## Features

### 1. **Reveal Answer Button**
- Appears in two scenarios:
  - When ML models return empty hints (not enough data)
  - When there's an error fetching ML hints
- Shows alongside "Try Again" button

### 2. **Confirmation Dialog**
- User must confirm before revealing the answer
- Dialog displays warning that this will end the current game
- Options: "Keep Guessing" or "Yes, Reveal Answer"

### 3. **Answer Display**
- Shows the correct character name in a highlighted card
- Gradient yellow/orange styling to make it stand out
- Sparkle animation for visual emphasis
- Encourages user to start a new game

## User Flow

1. User makes 4+ incorrect guesses
2. ML Hint System activates automatically
3. If ML models can't provide hints (insufficient data):
   - Error message appears: "Not enough clues available for AI hints yet"
   - Two buttons shown: "Try Again" and "Reveal Answer"
4. User clicks "Reveal Answer"
5. Confirmation dialog appears with warning
6. User confirms by clicking "Yes, Reveal Answer"
7. Correct character name is displayed in highlighted card
8. User can start a new game to try again

## Technical Implementation

### Frontend Components

#### MLHintSystem.tsx
- Added `correctAnswer` prop (string | null)
- Added `onRevealAnswer` callback prop
- New state: `showRevealDialog` and `answerRevealed`
- Imported AlertDialog from Radix UI
- Added "Reveal Answer" button in error and empty hints states
- Added confirmation dialog with AlertDialog component
- Added revealed answer display card

#### Index.tsx
- New state: `correctAnswer` (string | null)
- New handler: `handleRevealAnswer()` - calls API and sets correctAnswer
- Passes `correctAnswer` and `onRevealAnswer` to MLHintSystem

### Backend Routes

#### server/routes/game.ts
- New export: `handleRevealAnswer`
- Endpoint: POST `/api/game/reveal-answer`
- Returns: `{ success: true, character: Character }`
- Logs when answer is revealed for analytics

#### server/index.ts
- Imported `handleRevealAnswer`
- Registered route: `app.post("/api/game/reveal-answer", handleRevealAnswer)`

## Visual Design

### Reveal Answer Button
- Gradient background: Yellow (600) to Orange (600)
- Eye icon with "Reveal Answer" text
- Hover effect: Yellow (700) to Orange (700)

### Confirmation Dialog
- Dark gray background with yellow border
- Yellow warning icon and title
- Warning message about ending the game
- Two-button footer: "Keep Guessing" (gray) and "Yes, Reveal Answer" (gradient)

### Revealed Answer Card
- Gradient background: Yellow/Orange with 20% opacity
- Yellow border (2px)
- Pulsing sparkle icon
- Large character name (3xl font)
- Encouragement message

## API Endpoints

### POST /api/game/reveal-answer
**Headers:**
- `X-Session-Id`: User's session ID

**Response:**
```json
{
  "success": true,
  "character": {
    "id": "...",
    "name": "Character Name",
    "aliases": [...],
    "universe": "...",
    "quote": "...",
    "source": "...",
    "genre": "...",
    "imageUrl": "...",
    "characterImageUrl": "...",
    "attributes": {...}
  }
}
```

**Error Response:**
```json
{
  "error": "Character not found"
}
```

## Console Logging
- `[Reveal Answer] Session: {sessionId} revealed answer: {characterName}`

## Testing Steps

1. Start a new game
2. Make 4 incorrect guesses
3. Wait for ML Hint System to appear
4. If hints load: click a hint to test auto-fill
5. If no hints: click "Reveal Answer" button
6. Confirm in dialog by clicking "Yes, Reveal Answer"
7. Verify correct character name is displayed
8. Start a new game to test again

## Dependencies

### New Imports
- `Eye` icon from `lucide-react`
- `AlertTriangle` icon from `lucide-react`
- `AlertDialog` components from `@/components/ui/alert-dialog`

### Existing Dependencies
- Radix UI Alert Dialog (already installed)
- TailwindCSS for styling
- React hooks (useState)

## Future Enhancements

- [ ] Add animation when answer is revealed
- [ ] Track "gave up" statistics
- [ ] Show character image along with name
- [ ] Add confetti effect when answer is revealed
- [ ] Disable guessing after reveal
- [ ] Add "Share" button to share results
- [ ] Show fun facts about the revealed character
