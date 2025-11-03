import { useEffect, useState } from "react";

interface Particle {
  id: number;
  left: number;
  size: number;
  duration: number;
  delay: number;
  color: "red" | "white" | "gold" | "blue";
  blur: number;
  opacity: number;
}

export default function ParticlesBackground() {
  const [particles, setParticles] = useState<Particle[]>([]);

  useEffect(() => {
    const generateParticles = () => {
      const newParticles: Particle[] = [];
      const particleCount = 60; // Increased from 30 to 60

      for (let i = 0; i < particleCount; i++) {
        const colorRand = Math.random();
        let color: "red" | "white" | "gold" | "blue";
        
        // More variety in colors
        if (colorRand < 0.4) color = "red";
        else if (colorRand < 0.7) color = "white";
        else if (colorRand < 0.85) color = "gold";
        else color = "blue";

        newParticles.push({
          id: i,
          left: Math.random() * 100,
          size: Math.random() * 8 + 2, // Increased from 3+1 to 8+2
          duration: Math.random() * 15 + 10, // Faster movement (was 20+15)
          delay: Math.random() * 8,
          color: color,
          blur: Math.random() * 3 + 1, // Add blur effect
          opacity: Math.random() * 0.5 + 0.5, // Varied opacity
        });
      }

      setParticles(newParticles);
    };

    generateParticles();
  }, []);

  return (
    <div className="particles-background">
      {/* Add animated gradient overlay for extra visual interest */}
      <div className="particles-gradient-overlay" />
      
      {particles.map((particle) => (
        <div
          key={particle.id}
          className={`particle ${particle.color}`}
          style={{
            left: `${particle.left}%`,
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            filter: `blur(${particle.blur}px)`,
            opacity: particle.opacity,
            animation:
              Math.random() > 0.5
                ? `float-up-left ${particle.duration}s ease-in-out ${particle.delay}s infinite`
                : `float-up-right ${particle.duration}s ease-in-out ${particle.delay}s infinite`,
          }}
        />
      ))}
      
      {/* Add some larger glowing orbs for visual appeal */}
      {[...Array(5)].map((_, i) => (
        <div
          key={`orb-${i}`}
          className="particle-orb"
          style={{
            left: `${20 * i + 10}%`,
            animationDelay: `${i * 2}s`,
          }}
        />
      ))}
    </div>
  );
}
