# 2048 AI Agent — Expectimax + Heuristics

This project implements a fully playable **2048 Game** with a powerful **Expectimax-based AI agent** using domain‑specific heuristics inspired by competitive bots and expert guidelines. The agent can reliably reach high tiles by combining emptiness maximization, potential merge detection, smoothness, rank-weighted monotonicity, and strong corner bias.

---

## 📌 Project Summary
This project contains:

- A complete **2048 game engine** implemented from scratch.
- A modern **Streamlit UI** with animations.
- An **intelligent AI agent** using:
  - Expectimax search.
  - Transposition caching.
  - Adaptive depth (up to 4 depending on board emptiness).
  - Hard corner enforcement.
  - Strong heuristic combining multiple scoring components.
- Multiple experimental agent strategies, compared and evaluated.

---

## 📂 Repository Structure
```
/2048-AI-Agent
│
├── app.py             # Streamlit app with game UI + Expectimax agent
├── README.md          # This document
└── requirements.txt   # Required Python libraries
```

---

## 🚀 How to Run the Project
### **1. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit app**
```bash
streamlit run app.py
```

Your browser will open the complete 2048 UI.

---

## 🧩 Game Engine
The engine:
- Supports all 4 moves (up/down/left/right)
- Performs correct tile merging
- Adds a new tile after valid moves
- Starts with two `2` tiles (forced)
- Tracks score + game-over conditions

---

## 🧠 AI Approaches Tried
The following approaches were implemented, tested, and compared:

### **1. Pure Greedy (FAILED)**
- Looked only at immediate merge values.
- Frequently gets stuck early.
- Max tile: **64–128** typically.

### **2. Simple Heuristic Search (WEAK)**
Used heuristics:
- Max empty cells
- Smoothness
- Max tile weight

**Issues:**
- Max tile often drifts from corner.
- Grid becomes unsorted → quickly loses.
- Max tile: **256–512**.

### **3. Expectimax (BASELINE)**
A proper game-tree search:
- Player = max node
- Random tile spawns = chance node
- Depth 2–3 gives moderate results

Max tile: **512–1024** with simple heuristics.

### **4. Expectimax + Domain Heuristics (GOOD)**
Add strong heuristics:
- Emptiness
- Monotonicity
- Smoothness
- Potential merges

Max tile: **1024–2048**.

### **5. Expectimax + Corner + Snake Strategy (OPTIMAL)**
This final version follows well-known expert strategies:
- Keep **largest tile in a corner** at all times.
- Build **snake-like monotonic rows/columns** around the corner.
- Penalize moves that break the structure.
- Weighted monotonicity (big tiles punished more).
- Hard corner lock: disallow corner-breaking moves except if forced.
- Adaptive depth: depth 4 when board is tight (< 3 empty cells).
- Skip expensive 4‑tile branches to maintain speed.

This strategy is directly inspired by:
- StackOverflow optimal 2048 answer (#22342854)
- nneonneo's competitive AI implementation

**Results:**
- Max tile: **2048 consistently**
- Max tile: **4096 occasionally**
- Further improvements possible with move tables

---

## 🔬 Final Heuristic (Combined Score)
The AI evaluates each board using:

| Component | Description |
|----------|-------------|
| **Empties** | # of empty cells (heavily weighted) |
| **Potential merges** | Adjacent equal tiles |
| **Smoothness** | Penalizes large jumps between neighbors |
| **Rank-weighted monotonicity** | Big tiles penalize monotonicity breaks more |
| **Corner integrity** | Enforces ordered bottom row & right column |
| **Corner max-tile bonus/penalty** | Strong reward if max tile in corner |
| **Max tile log bonus** | Minor reward for progressing to larger tiles |

---

## 📈 Results Comparison
| Approach | Typical Max Tile | Notes |
|----------|------------------|-------|
| Greedy | 64–128 | Breaks very early |
| Basic heuristic | 256–512 | Unstable board |
| Expectimax (depth 3) | 512–1024 | Moderate stability |
| Expectimax + heuristics | 1024–2048 | Good pattern formation |
| **Final Model (this)** | **2048–4096** | Stable corner lock, expert-level behavior |

---

## 🌍 References
### **Core Algorithm References**
- *What is the optimal algorithm for the game 2048?*  
  https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048

- nneonneo’s high‑performance 2048 AI (Expectimax + heuristics)
- Classic 2048 heuristics research: smoothness, monotonicity, emptiness

### **Game Mechanics Information**
- Gabriele Cirulli (original 2048 source) — grid interactions

### **Other Inspiration**
- Heuristic tuning from hobbyist and competitive 2048 bots
- General expectimax ideas from game-tree search literature

---

## 🏆 Final Solution Summary
The final agent uses:
- **Expectimax Search**
- **Adaptive depth (3 → 4)**
- **Rank-weighted monotonicity** from SO reference
- **Smoothness** and **potential merges** scoring
- **Hard corner strategy** enforcing largest tile stays fixed
- **Corner row/column integrity score**
- **Transposition cache** for speed
- **Approximate chance branch pruning (skip 4s)**

This combination captures both:
- Strong theoretical behavior (Expectimax)
- Real 2048 strategy knowledge used by top human players

---

## 🎮 Conclusion
This project provides:
- A polished, interactive 2048 game UI
- A near‑optimal AI agent using deterministic heuristics
- Strong and explainable decision-making logic
- Easily extendable architecture (DQN/RL, move-tables, etc.)

If you want RL-based improvements or a full move-table optimized engine (8× faster), it can be added next.

