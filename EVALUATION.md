# Bewertung: KI-Training-Ansatz für WoW-Bot

## Was gut funktioniert

Das Grundkonzept ist solide: Ein C++-Modul als Bridge zum Gameserver, Gymnasium-kompatibles Environment, und PPO als Algorithmus. Das ist ein vernünftiger Tech-Stack. Die Idee, Bots serverseitig zu spawnen (ohne echten Client), ist clever und vermeidet viel Overhead. Die NPC-Memory und der Hybrid-Ansatz in `auto_grind.py` zeigen gutes Systemverständnis.

## Kernprobleme, die sinnvolles Training verhindern

### 1. Echtzeit-Kopplung — der größte Flaschenhals

Das Training läuft an die Gameserver-Geschwindigkeit gebunden (400ms State-Updates). Bei 10.000 Timesteps dauert ein Trainingslauf ~67 Minuten Wanduhrzeit — und liefert praktisch kein brauchbares Lernergebnis. PPO braucht typischerweise **Millionen** Timesteps, um für Aufgaben dieser Komplexität zu konvergieren. Bei der aktuellen Architektur wären das **Monate** Echtzeit.

Zum Vergleich: OpenAI Five (Dota 2) nutzte das Äquivalent von 180 Jahren Spielzeit pro Tag durch massiv beschleunigte Simulation. Selbst einfachere RL-Projekte brauchen 10-100x Echtzeit-Beschleunigung.

### 2. Die Override-Logik untergräbt das RL

Die ~50 Zeilen Override-Logik in `wow_env.py:210-278` überschreiben RL-Entscheidungen in den meisten interessanten Situationen:

- Vendor-Navigation: hardcoded
- Aggro-Recovery: hardcoded
- Cast-Schutz: hardcoded
- Loot-Automatik: hardcoded
- Range-Management: hardcoded
- Heal-Sperre: hardcoded

Was bleibt dem RL-Agent übrig? Im Wesentlichen nur: "Wann Smite casten, wann drehen, wann vorwärts gehen" — und selbst da wird vieles gefiltert. Das ist kein RL-Training, das ist ein Scripted-Bot mit einem RL-Feigenblatt.

### 3. Observation Space viel zu klein (10 Floats)

Der Agent hat extrem eingeschränkte Wahrnehmung:

- **Kein Wissen über mehrere Mobs** — nur das aktuelle Target
- **Keine Cooldown-Info** — weiß nicht, ob Smite bereit ist
- **Keine Umgebungsinfo** — keine Terrain-/Hindernisdaten
- **Kein Gedächtnis** — MlpPolicy hat kein LSTM, jeder Tick ist isoliert
- **Index 8 ist immer 0** — verschwendeter Dimension

Mit 10 Floats kann der Agent die Spielsituation nicht ausreichend erfassen, um sinnvolle Strategien zu lernen.

### 4. Action Space zu grob (Discrete(9))

- Drehung nur in festen 0.5-rad-Schritten (~28°) — viel zu grob für Positionierung
- Vorwärtsbewegung nur 3 Units — kein variables Movement
- Nur 2 Spells — ein Priest hat 20+ relevante Fähigkeiten
- Kein Strafing, Kiting, Rückwärtslaufen
- Kein `move_to` als Action verfügbar (nur in Overrides)

### 5. Reward-Design ist problematisch

| Signal | Wert | Problem |
|---|---|---|
| Step-Penalty | -0.01 | 10.000x kleiner als Kill-Reward |
| Kill (XP) | +100 bis +300 | Sehr selten, dominiert alles |
| Level-Up | +2000 | Terminiert Episode UND resettet Level → verwirrender Loop |
| Tod | -100 | Asymmetrisch zum Kill-Reward |
| Smite + Target | +0.5 | Motiviert Spammen statt strategisches Timing |
| Bewegen im Kampf | -0.5 / +0.6 | Widersprüchlich (Strafe für Bewegung, Bonus fürs Drehen) |

Die Reward-Skalen sind über 5 Größenordnungen verteilt. Der Agent wird primär lernen, Kills zu maximieren, und alle Feinheiten (Positioning, Heal-Timing, Mana-Management) werden im Rauschen untergehen.

### 6. nearby_mobs Cache-Bug

In `AIControllerHook.cpp` wird `_cachedNearbyMobsJson` nur für den **ersten** Online-Spieler berechnet (durch ein `break`). Alle 5 Bots bekommen dieselbe Mob-Liste — auch wenn sie auf verschiedenen Teilen der Map stehen. Das korrumpiert die Observations für 4 von 5 Bots.

### 7. 10.000 Timesteps sind nichts

Mit `n_steps=128` und 5 Envs ergibt das `10000 / (128 * 5) ≈ 15` Policy-Updates. PPO braucht typischerweise tausende Updates, um zu konvergieren. Das aktuelle Setup kann unmöglich etwas Sinnvolles lernen.

## Verbesserungsvorschläge (nach Priorität)

### Stufe 1: Machbar mit dem aktuellen Setup

**a) nearby_mobs Bug fixen** — Pro-Bot Mob-Scanning statt globaler Cache. Ohne das sind Multi-Bot-Trainings sinnlos.

**b) Timesteps auf 500k-1M hochsetzen** — Selbst bei Echtzeit-Kopplung braucht PPO mehr Daten. Mit 5 Bots parallel und einem dedizierten Server ist das zumindest über Nacht machbar.

**c) Reward normalisieren** — Alle Rewards auf ähnliche Skalen bringen (z.B. -1 bis +1 Bereich). Potential-based Reward Shaping verwenden statt roher Werte.

**d) Observation Space erweitern auf ~30-50 Dimensionen:**

- Top-3 nearby Mobs (je: Distanz, Winkel, HP, attackable) = +12
- Spell-Cooldowns (Smite, Heal) = +2
- Eigener Movement-State (speed, ist_laufend) = +2
- Letzte 3-5 Actions (Action-History) = +5
- Mana als absolute Zahl (für Cast-Planung) = +1

**e) Override-Logik schrittweise entfernen** — Stattdessen als zusätzliche Reward-Signale formulieren ("Wenn du im Kampf ohne Target bist, kriege Reward fürs Targeten, nicht Strafe fürs Nicht-Targeten").

### Stufe 2: Architektur-Verbesserungen

**f) Simulation beschleunigen.** Drei Optionen:

1. **AzerothCore im Fast-Forward**: Server-Tick-Rate hochdrehen (schwer, aber möglich — der Timer in `OnUpdate` ist nur ein Delay, die Gamelogik ist frame-unabhängig)
2. **Lightweight Python-Simulation**: Ein vereinfachtes Kampfmodell in reinem Python (HP, Mana, Spell-Damage, Cooldowns, Position). Training in der Sim, dann Transfer zum echten Server. Das wäre 1000-10.000x schneller.
3. **Offline RL**: Daten sammeln mit dem Scripted-Bot, dann offline mit CQL/IQL trainieren.

**g) Recurrent Policy (LSTM)** statt MlpPolicy — damit der Agent über mehrere Ticks hinweg planen kann. In SB3: `RecurrentPPO` aus `sb3-contrib`.

**h) Hierarchical RL:**

- **High-Level Policy** (alle 5-10s): Wählt Strategie (Kämpfen, Navigieren, Heilen, Looten, Vendor)
- **Low-Level Policy** (jeden Tick): Führt die gewählte Strategie aus

Das ersetzt die Override-Logik durch gelernte Hierarchie.

### Stufe 3: Für wirklich ambitioniertes Training

**i) Imitation Learning als Kickstart:**

- Menschliche Gameplay-Daten sammeln (oder den gut funktionierenden Scripted-Bot in `test_multibot.py` als Expert)
- Behavioral Cloning als Vortraining
- Dann PPO für Fine-Tuning

**j) Self-Play / Population-Based Training:**

- Viele Agenten mit leicht verschiedenen Hyperparametern parallel trainieren
- Die besten überleben, schlechte werden durch Mutationen der guten ersetzt

**k) Multi-Klassen-Support:**

- Verschiedene Klassen mit verschiedenen Spell-Sets
- Transfer Learning zwischen Klassen

## Fazit

Das aktuelle Setup ist ein **funktionierender Prototyp**, der beweist, dass die Infrastruktur (C++↔Python Bridge, Bot-Spawning, State-Streaming) grundsätzlich funktioniert. Aber als RL-Trainings-Pipeline ist es nicht skalierbar:

- **Echtzeit-Kopplung** macht Training ~1000x zu langsam
- **10 Observations + 9 Actions** sind zu wenig für sinnvolles Lernen
- **Override-Logik** nimmt dem Agent die interessantesten Entscheidungen ab
- **10k Timesteps** sind ~1000x zu wenig
- **Reward-Skalen** über 5 Größenordnungen verhindern stabiles Lernen

Der pragmatischste nächste Schritt wäre: **Eine vereinfachte Python-Combat-Simulation bauen**, die denselben Observation/Action-Space hat, aber 10.000x schneller läuft. Dort trainieren, dann das Modell auf den echten Server transferieren. Das ist der Ansatz, den quasi alle erfolgreichen Game-RL-Projekte nutzen (Sim-to-Real Transfer).
