import { Character } from "@shared/api";

/**
 * Character database for the guessing game
 * Contains popular superheroes from Marvel and DC universes
 * Using placeholder images that will work - replace with actual superhero images
 */
export const CHARACTERS: Character[] = [
  // Marvel Characters
  {
    id: "iron-man",
    name: "Iron Man",
    aliases: ["Tony Stark", "Stark", "Tony"],
    universe: "Marvel",
    quote: "I am Iron Man.",
    source: "Iron Man",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/35860/forge-craft-hot-form.jpg", // Iron/metal texture (clue)
    characterImageUrl: "https://images.pexels.com/photos/11876188/pexels-photo-11876188.jpeg", // Iron Man suit/armor
    attributes: {
      alignment: "hero",
      powers: ["Genius intellect", "Powered armor suit", "Flight", "Energy weapons"],
      team: "The Avengers",
      firstAppearance: "Tales of Suspense #39 (1963)",
    },
  },
  {
    id: "spider-man",
    name: "Spider-Man",
    aliases: ["Peter Parker", "Spidey", "Web-Slinger", "Peter", "Parker"],
    universe: "Marvel",
    quote: "With great power comes great responsibility.",
    source: "Spider-Man",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/378570/pexels-photo-378570.jpeg?_gl=1*mgkx14*_ga*MTcyOTM0NTA5MC4xNzYyMTU4MTM3*_ga_8JE65Q40S6*czE3NjIxNTgxMzckbzEkZzEkdDE3NjIxNTgyNjEkajI2JGwwJGgw", // Spider and web (clue)
    characterImageUrl: "https://images.pexels.com/photos/5691158/pexels-photo-5691158.jpeg", // Spider-Man in costume
    attributes: {
      alignment: "hero",
      powers: ["Wall-crawling", "Spider-sense", "Superhuman strength", "Web-shooters"],
      team: "The Avengers",
      firstAppearance: "Amazing Fantasy #15 (1962)",
    },
  },
  {
    id: "captain-america",
    name: "Captain America",
    aliases: ["Steve Rogers", "Cap", "The First Avenger", "Steve", "Rogers"],
    universe: "Marvel",
    quote: "I can do this all day.",
    source: "Captain America: The First Avenger",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/973049/pexels-photo-973049.jpeg?_gl=1*1edr23w*_ga*MTcyOTM0NTA5MC4xNzYyMTU4MTM3*_ga_8JE65Q40S6*czE3NjIxNTgxMzckbzEkZzEkdDE3NjIxNTgxNDUkajUyJGwwJGgw", // American flag/stars and stripes (clue)
    characterImageUrl: "https://images.pexels.com/photos/10682514/pexels-photo-10682514.jpeg", // Captain America with shield
    attributes: {
      alignment: "hero",
      powers: ["Super soldier serum", "Enhanced strength", "Tactical genius", "Vibranium shield"],
      team: "The Avengers",
      firstAppearance: "Captain America Comics #1 (1941)",
    },
  },
  {
    id: "thor",
    name: "Thor",
    aliases: ["Thor Odinson", "God of Thunder", "Odinson"],
    universe: "Marvel",
    quote: "I am Thor, son of Odin!",
    source: "Thor",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/40721/egg-hammer-threaten-violence-40721.jpeg", // Lightning/thunder storm (clue)
    characterImageUrl: "https://media.istockphoto.com/id/1147508579/photo/man-in-cosplaying-thor-isolated-on-white-studio-background.jpg?s=2048x2048&w=is&k=20&c=KJy3xpCa7EHpKcONnwjZIAuDxHbBFPQTy8urCoJAUl0=", // Thor with hammer
    attributes: {
      alignment: "hero",
      powers: ["Superhuman strength", "Lightning control", "Mjolnir", "Immortality"],
      team: "The Avengers",
      firstAppearance: "Journey into Mystery #83 (1962)",
    },
  },
  {
    id: "hulk",
    name: "Hulk",
    aliases: ["Bruce Banner", "The Incredible Hulk", "Green Goliath", "Bruce", "Banner"],
    universe: "Marvel",
    quote: "That's my secret, Cap. I'm always angry.",
    source: "The Avengers",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/2338015/pexels-photo-2338015.jpeg", // Green/smashing fist (clue)
    characterImageUrl: "https://images.pexels.com/photos/12689085/pexels-photo-12689085.jpeg", // Hulk
    attributes: {
      alignment: "hero",
      powers: ["Unlimited strength", "Regeneration", "Durability", "Gamma radiation"],
      team: "The Avengers",
      firstAppearance: "The Incredible Hulk #1 (1962)",
    },
  },
  {
    id: "black-widow",
    name: "Black Widow",
    aliases: ["Natasha Romanoff", "Natasha", "Romanoff"],
    universe: "Marvel",
    quote: "I've got red in my ledger.",
    source: "The Avengers",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/16326004/pexels-photo-16326004.jpeg", // Black widow spider (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/f/f6/Scarlett_Johansson_as_Black_Widow.jpg", // Black Widow character
    attributes: {
      alignment: "hero",
      powers: ["Master spy", "Expert martial artist", "Weapons expert", "Enhanced physiology"],
      team: "The Avengers",
      firstAppearance: "Tales of Suspense #52 (1964)",
    },
  },
  {
    id: "doctor-strange",
    name: "Doctor Strange",
    aliases: ["Stephen Strange", "Sorcerer Supreme", "Stephen", "Strange"],
    universe: "Marvel",
    quote: "We're in the endgame now.",
    source: "Doctor Strange",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/263337/pexels-photo-263337.jpeg", // Mystical portal/magic circle (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/1/18/Benedict_Cumberbatch_as_Doctor_Strange.jpeg", // Doctor Strange with cape
    attributes: {
      alignment: "hero",
      powers: ["Sorcery", "Time manipulation", "Astral projection", "Mystic arts"],
      team: "The Avengers",
      firstAppearance: "Strange Tales #110 (1963)",
    },
  },
  {
    id: "black-panther",
    name: "Black Panther",
    aliases: ["T'Challa", "King of Wakanda", "TChalla"],
    universe: "Marvel",
    quote: "Wakanda forever!",
    source: "Black Panther",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/260024/pexels-photo-260024.jpeg", // Black panther animal (clue)
    characterImageUrl: "https://static.wikia.nocookie.net/marvelcinematicuniverse/images/9/9d/T%27Challa_Infobox.jpg/revision/latest?cb=20231024023619", // Black Panther suit
    attributes: {
      alignment: "hero",
      powers: ["Enhanced strength", "Vibranium suit", "Genius intellect", "Master tactician"],
      team: "The Avengers",
      firstAppearance: "Fantastic Four #52 (1966)",
    },
  },
  {
    id: "thanos",
    name: "Thanos",
    aliases: ["The Mad Titan", "Mad Titan"],
    universe: "Marvel",
    quote: "I am inevitable.",
    source: "Avengers: Infinity War",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/1147946/pexels-photo-1147946.jpeg", // Infinity stones/gems (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/7/7b/Josh_Brolin_as_Thanos.jpeg", // Thanos
    attributes: {
      alignment: "villain",
      powers: ["Superhuman strength", "Infinity Gauntlet", "Genius intellect", "Immortality"],
      firstAppearance: "The Invincible Iron Man #55 (1973)",
    },
  },
  {
    id: "loki",
    name: "Loki",
    aliases: ["God of Mischief", "Loki Laufeyson", "Laufeyson"],
    universe: "Marvel",
    quote: "I am burdened with glorious purpose.",
    source: "The Avengers",
    genre: "Superhero Action",
    imageUrl: "https://cdn.britannica.com/65/166665-050-50B725A9/binding-Fenrir.jpg", // Horns/mischievous mask (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/commons/4/4a/Tom_Hiddleston_by_Gage_Skidmore.jpg", // Loki with horns
    attributes: {
      alignment: "villain",
      powers: ["Sorcery", "Shapeshifting", "Illusions", "Superhuman strength"],
      firstAppearance: "Journey into Mystery #85 (1962)",
    },
  },
  {
    id: "scarlet-witch",
    name: "Scarlet Witch",
    aliases: ["Wanda Maximoff", "Wanda"],
    universe: "Marvel",
    quote: "You took everything from me.",
    source: "WandaVision",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/9665491/pexels-photo-9665491.jpeg", // Red magic/scarlet energy (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/e/ec/Scarlet_Witch_Various_incarnations_2021.jpg", // Scarlet Witch
    attributes: {
      alignment: "hero",
      powers: ["Reality manipulation", "Chaos magic", "Telekinesis", "Mind control"],
      team: "The Avengers",
      firstAppearance: "The X-Men #4 (1964)",
    },
  },
  {
    id: "vision",
    name: "Vision",
    aliases: ["The Vision"],
    universe: "Marvel",
    quote: "A thing isn't beautiful because it lasts.",
    source: "Avengers: Age of Ultron",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/20419740/pexels-photo-20419740.jpeg", // Mind stone/yellow gem (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/f/fc/Paul_Bettany_as_Vision.jpg", // Vision android
    attributes: {
      alignment: "hero",
      powers: ["Superhuman strength", "Flight", "Density control", "Mind Stone"],
      team: "The Avengers",
      firstAppearance: "The Avengers #57 (1968)",
    },
  },
  {
    id: "ant-man",
    name: "Ant-Man",
    aliases: ["Scott Lang", "Scott"],
    universe: "Marvel",
    quote: "I do some dumb things, and the people I love the most pay the price.",
    source: "Ant-Man",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/1104974/pexels-photo-1104974.jpeg", // Ants (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/1/12/Ant-Man_%28film%29_poster.jpg", // Ant-Man in costume
    attributes: {
      alignment: "hero",
      powers: ["Size manipulation", "Ant communication", "Enhanced strength when small"],
      team: "The Avengers",
      firstAppearance: "Tales to Astonish #35 (1962)",
    },
  },

  // DC Characters
  {
    id: "batman",
    name: "Batman",
    aliases: ["Bruce Wayne", "The Dark Knight", "Caped Crusader", "Bruce", "Wayne"],
    universe: "DC",
    quote: "I'm vengeance.",
    source: "Batman Begins",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/257970/pexels-photo-257970.jpeg", // Bat animal (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/commons/e/e4/San_Diego_Comic-Con_2024_Masquerade_-_Cosplay_of_Batman_3.jpg", // Batman in costume
    attributes: {
      alignment: "hero",
      powers: ["Genius intellect", "Master detective", "Martial arts expert", "Wealth and technology"],
      team: "Justice League",
      firstAppearance: "Detective Comics #27 (1939)",
    },
  },
  {
    id: "superman",
    name: "Superman",
    aliases: ["Clark Kent", "Kal-El", "Man of Steel", "Clark", "Kent"],
    universe: "DC",
    quote: "I'm here to fight for truth, justice, and the American way.",
    source: "Superman",
    genre: "Superhero Action",
    imageUrl: "https://media.istockphoto.com/id/886851320/photo/kryptonite-crystal.jpg?s=2048x2048&w=is&k=20&c=ZY4ky_H7f5QHFurOJE3xaFbmcImiNDCh4qP8NS7Jrzw=", // Kryptonite (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/d/d6/Superman_Man_of_Steel.jpg", // Superman flying
    attributes: {
      alignment: "hero",
      powers: ["Flight", "Super strength", "Heat vision", "Invulnerability", "Super speed"],
      team: "Justice League",
      firstAppearance: "Action Comics #1 (1938)",
    },
  },
  {
    id: "wonder-woman",
    name: "Wonder Woman",
    aliases: ["Diana Prince", "Amazon Princess", "Diana"],
    universe: "DC",
    quote: "I will fight for those who cannot fight for themselves.",
    source: "Wonder Woman",
    genre: "Superhero Action",
    imageUrl: "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=500&h=500&fit=crop", // Golden lasso/tiara (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/6/6b/Wonder_Woman_750.jpg", // Wonder Woman with lasso
    attributes: {
      alignment: "hero",
      powers: ["Superhuman strength", "Flight", "Lasso of Truth", "Combat mastery", "Immortality"],
      team: "Justice League",
      firstAppearance: "All Star Comics #8 (1941)",
    },
  },
  {
    id: "flash",
    name: "The Flash",
    aliases: ["Barry Allen", "Fastest Man Alive", "Barry", "Allen", "Flash"],
    universe: "DC",
    quote: "Life is locomotion. If you're not moving, you're not living.",
    source: "The Flash",
    genre: "Superhero Action",
    imageUrl: "https://images.pexels.com/photos/371838/pexels-photo-371838.jpeg", // Lightning bolt/speed (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/b/b7/Flash_%28Barry_Allen%29.png", // The Flash running
    attributes: {
      alignment: "hero",
      powers: ["Super speed", "Time travel", "Speed Force", "Accelerated healing"],
      team: "Justice League",
      firstAppearance: "Showcase #4 (1956)",
    },
  },
  {
    id: "aquaman",
    name: "Aquaman",
    aliases: ["Arthur Curry", "King of Atlantis", "Arthur"],
    universe: "DC",
    quote: "My father was a lighthouse keeper. My mother was a queen.",
    source: "Aquaman",
    genre: "Superhero Action",
    imageUrl: "https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=500&h=500&fit=crop", // Ocean/trident (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/e/ed/Aquaman_%28film%29_poster.jpg", // Aquaman with trident
    attributes: {
      alignment: "hero",
      powers: ["Underwater breathing", "Superhuman strength", "Marine telepathy", "Trident of Poseidon"],
      team: "Justice League",
      firstAppearance: "More Fun Comics #73 (1941)",
    },
  },
  {
    id: "cyborg",
    name: "Cyborg",
    aliases: ["Victor Stone", "Victor"],
    universe: "DC",
    quote: "I'm not broken.",
    source: "Justice League",
    genre: "Superhero Action",
    imageUrl: "https://images.unsplash.com/photo-1535378620166-273708d44e4c?w=500&h=500&fit=crop", // Robotic/circuits (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/5/58/Cyborg_%28Victor_Stone%29.jpg", // Cyborg character
    attributes: {
      alignment: "hero",
      powers: ["Technopathy", "Superhuman strength", "Weapons systems", "Computer interface"],
      team: "Justice League",
      firstAppearance: "DC Comics Presents #26 (1980)",
    },
  },
  {
    id: "joker",
    name: "The Joker",
    aliases: ["Jack Napier", "Joker"],
    universe: "DC",
    quote: "Why so serious?",
    source: "The Dark Knight",
    genre: "Superhero Action",
    imageUrl: "https://images.unsplash.com/photo-1551269901-5c5e14c25df7?w=500&h=500&fit=crop", // Playing cards/joker card (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/9/90/HeathJoker.png", // Joker face
    attributes: {
      alignment: "villain",
      powers: ["Criminal mastermind", "Unpredictability", "Expert chemist", "Pain tolerance"],
      firstAppearance: "Batman #1 (1940)",
    },
  },
  {
    id: "harley-quinn",
    name: "Harley Quinn",
    aliases: ["Harleen Quinzel", "Dr. Quinzel", "Harley", "Harleen"],
    universe: "DC",
    quote: "We're bad guys. It's what we do.",
    source: "Suicide Squad",
    genre: "Superhero Action",
    imageUrl: "https://images.unsplash.com/photo-1571216776960-faa251eea1f7?w=500&h=500&fit=crop", // Baseball bat/hammer (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/5/56/SuicideSquadHarleyQuinn.jpg", // Harley Quinn
    attributes: {
      alignment: "anti-hero",
      powers: ["Gymnast", "Combat skills", "Immunity to toxins", "Unpredictability"],
      team: "Suicide Squad",
      firstAppearance: "Batman: The Animated Series (1992)",
    },
  },
  {
    id: "green-lantern",
    name: "Green Lantern",
    aliases: ["Hal Jordan", "Emerald Knight", "Hal", "Jordan"],
    universe: "DC",
    quote: "In brightest day, in blackest night, no evil shall escape my sight.",
    source: "Green Lantern",
    genre: "Superhero Action",
    imageUrl: "https://images.unsplash.com/photo-1513436539083-9d2127e742f1?w=500&h=500&fit=crop", // Green lantern/light (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/7/75/Green_Lantern_%28John_Stewart%29.png", // Green Lantern with ring
    attributes: {
      alignment: "hero",
      powers: ["Power ring", "Energy constructs", "Flight", "Force fields"],
      team: "Justice League",
      firstAppearance: "Showcase #22 (1959)",
    },
  },
  {
    id: "shazam",
    name: "Shazam",
    aliases: ["Billy Batson", "Captain Marvel", "Billy"],
    universe: "DC",
    quote: "Say my name!",
    source: "Shazam!",
    genre: "Superhero Action",
    imageUrl: "https://upload.wikimedia.org/wikipedia/en/d/dd/WizardShazamGaryFrank.jpg", // Lightning/thunder bolt (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/d/d5/Billy_Batson%2C_Shazam_Captain_Marvel_%28Modern%29.png", // Shazam with lightning
    attributes: {
      alignment: "hero",
      powers: ["Superhuman strength", "Flight", "Lightning manipulation", "Magic"],
      team: "Justice League",
      firstAppearance: "Whiz Comics #2 (1940)",
    },
  },
  {
    id: "deadpool",
    name: "Deadpool",
    aliases: ["Wade Wilson", "Merc with a Mouth", "Wade", "Wilson"],
    universe: "Marvel",
    quote: "Maximum effort!",
    source: "Deadpool",
    genre: "Superhero Action",
    imageUrl: "https://previews.123rf.com/images/viktorijareut/viktorijareut1508/viktorijareut150800330/44024673-two-crossed-katana-swords-vector-illustration-samurai-sword-traditional-asian-ninja-weapon.jpg", // Katanas/swords (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/2/23/Deadpool_%282016_poster%29.png", // Deadpool in costume
    attributes: {
      alignment: "anti-hero",
      powers: ["Regeneration", "Expert marksman", "Sword master", "Fourth wall breaking"],
      firstAppearance: "The New Mutants #98 (1991)",
    },
  },
  {
    id: "wolverine",
    name: "Wolverine",
    aliases: ["Logan", "James Howlett", "Weapon X"],
    universe: "Marvel",
    quote: "I'm the best there is at what I do, but what I do isn't very nice.",
    source: "X-Men",
    genre: "Superhero Action",
    imageUrl: "https://images.unsplash.com/photo-1585411241865-c234229c37b7?w=500&h=500&fit=crop", // Wolverine animal/claws (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/d/d3/Wolverine_%28circa_2024%29.jpg", // Wolverine with claws
    attributes: {
      alignment: "hero",
      powers: ["Regeneration", "Adamantium claws", "Enhanced senses", "Longevity"],
      team: "X-Men",
      firstAppearance: "The Incredible Hulk #180 (1974)",
    },
  },
  {
    id: "storm",
    name: "Storm",
    aliases: ["Ororo Munroe", "Ororo"],
    universe: "Marvel",
    quote: "Do you know what happens to a toad when it's struck by lightning?",
    source: "X-Men",
    genre: "Superhero Action",
    imageUrl: "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=500&h=500&fit=crop", // Storm clouds/weather (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/3/34/Storm_%28Ororo_Munroe%29.png", // Storm with white hair
    attributes: {
      alignment: "hero",
      powers: ["Weather manipulation", "Flight", "Lightning control", "Wind generation"],
      team: "X-Men",
      firstAppearance: "Giant-Size X-Men #1 (1975)",
    },
  },
  {
    id: "cyclops",
    name: "Cyclops",
    aliases: ["Scott Summers", "Scott"],
    universe: "Marvel",
    quote: "To me, my X-Men!",
    source: "X-Men",
    genre: "Superhero Action",
    imageUrl: "https://media.istockphoto.com/id/1008655682/vector/creative-vector-illustration-of-laser-security-beam-isolated-on-transparent-background-art.jpg?s=1024x1024&w=is&k=20&c=6creKAp5tr5asoiOnokGgSt3TYx5FwMl3aSlLug7UAg=", // Red laser/eye beam (clue)
    characterImageUrl: "https://upload.wikimedia.org/wikipedia/en/e/e9/Cyclops_%28Scott_Summers_circa_2019%29.png", // Cyclops with visor
    attributes: {
      alignment: "hero",
      powers: ["Optic blast", "Tactical genius", "Energy projection"],
      team: "X-Men",
      firstAppearance: "The X-Men #1 (1963)",
    },
  },
];

/**
 * Get a character by ID
 */
export function getCharacterById(id: string): Character | undefined {
  return CHARACTERS.find((char) => char.id === id);
}

/**
 * Get a character by name (case-insensitive, checks aliases too)
 */
export function getCharacterByName(name: string): Character | undefined {
  const normalizedName = name.toLowerCase().trim();
  return CHARACTERS.find(
    (char) =>
      char.name.toLowerCase() === normalizedName ||
      char.aliases.some((alias) => alias.toLowerCase() === normalizedName)
  );
}

/**
 * Get all character names for autocomplete
 */
export function getAllCharacterNames(): Array<{ id: string; name: string; aliases: string[] }> {
  return CHARACTERS.map((char) => ({
    id: char.id,
    name: char.name,
    aliases: char.aliases,
  }));
}

/**
 * Select a character for a specific date (deterministic based on date)
 */
export function getDailyCharacter(date: string): Character {
  // Use date as seed for deterministic "random" selection
  const dateHash = date.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const index = dateHash % CHARACTERS.length;
  return CHARACTERS[index];
}

/**
 * Get a truly random character
 */
export function getRandomCharacter(): Character {
  const randomIndex = Math.floor(Math.random() * CHARACTERS.length);
  return CHARACTERS[randomIndex];
}
