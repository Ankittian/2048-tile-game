import streamlit as st
import random
import copy
import math
import numpy as np
from typing import List, Tuple, Optional


# Game engine 

class Game2048:
    def __init__(self, size=4, seed=None):
        self.size = size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    def reset(self):
        self.grid = [[0]*self.size for _ in range(self.size)]
        self.score = 0
        self._add_random_tile(force_two=True)
        self._add_random_tile(force_two=True)
        return self.grid

    def clone(self):
        g = Game2048(self.size)
        g.grid = copy.deepcopy(self.grid)
        g.score = self.score
        return g

    def _add_random_tile(self, force_two=False):
        empties = [(r,c) for r in range(self.size) for c in range(self.size) if self.grid[r][c]==0]
        if not empties:
            return False
        r,c = random.choice(empties)
        self.grid[r][c] = 2 if force_two else (4 if random.random() < 0.1 else 2)
        return True

    def can_move(self):
        if any(self.grid[r][c]==0 for r in range(self.size) for c in range(self.size)):
            return True
        for r in range(self.size):
            for c in range(self.size-1):
                if self.grid[r][c] == self.grid[r][c+1]:
                    return True
        for c in range(self.size):
            for r in range(self.size-1):
                if self.grid[r][c] == self.grid[r+1][c]:
                    return True
        return False

    def move(self, direction: str):
        """
        Standard 2048 move: compress -> merge -> compress applied per row/col.
        Returns (moved: bool, points_gained: int)
        """
        moved = False
        total_pts = 0
        N = self.size

        def compress(row):
            new_row = [v for v in row if v != 0]
            new_row += [0] * (N - len(new_row))
            return new_row

        def merge(row):
            pts = 0
            for i in range(N-1):
                if row[i] != 0 and row[i] == row[i+1]:
                    row[i] *= 2
                    pts += row[i]
                    row[i+1] = 0
            return row, pts

        for i in range(N):
            if direction in ("left","right"):
                row = self.grid[i][:]
                if direction == "right":
                    row.reverse()
                row = compress(row)
                row, pts = merge(row)
                row = compress(row)
                if direction == "right":
                    row.reverse()
                if row != self.grid[i]:
                    moved = True
                    self.grid[i] = row
                    total_pts += pts
            else:
                col = [self.grid[r][i] for r in range(N)]
                if direction == "down":
                    col.reverse()
                col = compress(col)
                col, pts = merge(col)
                col = compress(col)
                if direction == "down":
                    col.reverse()
                if col != [self.grid[r][i] for r in range(N)]:
                    moved = True
                    for r in range(N):
                        self.grid[r][i] = col[r]
                    total_pts += pts

        if moved:
            self.score += total_pts
            self._add_random_tile()

        return moved, total_pts


# Snake order & utility helpers 

SNAKE_ORDER = [
    (3,3),(3,2),(3,1),(3,0),
    (2,0),(2,1),(2,2),(2,3),
    (1,3),(1,2),(1,1),(1,0),
    (0,0),(0,1),(0,2),(0,3)
]

def grid_to_array(grid: List[List[int]]) -> np.ndarray:
    return np.array(grid, dtype=int)

def snake_sequence_values(a: np.ndarray) -> List[int]:
    return [int(a[r,c]) for (r,c) in SNAKE_ORDER]

def snake_violations(a: np.ndarray) -> int:
    vals = snake_sequence_values(a)
    violations = 0
    for i in range(len(vals)-1):
        if vals[i] < vals[i+1]:
            violations += 1
    return violations

def tile_positions(a: np.ndarray, value: int) -> List[Tuple[int,int]]:
    coords = list(zip(*np.where(a==value)))
    return coords

def manhattan(p1: Tuple[int,int], p2: Tuple[int,int]) -> int:
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def immediate_merge_max_value(grid: List[List[int]]) -> int:
    a = grid_to_array(grid)
    best = 0
    for r in range(4):
        for c in range(3):
            if a[r,c] != 0 and a[r,c] == a[r,c+1]:
                best = max(best, a[r,c]*2)
    for c in range(4):
        for r in range(3):
            if a[r,c] != 0 and a[r,c] == a[r+1,c]:
                best = max(best, a[r,c]*2)
    return best

# Heuristics

def strict_tile_alignment_penalty(a: np.ndarray) -> float:
    """
    Hard enforcement: each snake pair must be non-increasing.
    For each violation apply huge penalty (very large number).
    Also check whether 'expected' positions for top tiles are respected.
    """
    violations = snake_violations(a)
    penalty = -3200.0 * violations

    # We penalize if major tiles (>= 128) are not in top snake slots toward corner.
    maxv = int(a.max()) if a.size>0 else 0
    # expected positions for descending big tiles: assign first few snake slots
    big_thresholds = [maxv, maxv//2, maxv//4, maxv//8]
    # map values to nearest snake index
    for t in big_thresholds:
        if t <= 0: continue
        coords = tile_positions(a, t)
        if not coords:
            # if tile is missing (already merged), no penalty
            continue
        # pick top-left-most coordinate of that tile occurrence and check if it lies in early snake positions
        r,c = coords[0]
        snake_index = SNAKE_ORDER.index((3,3))  # base index 0
        # find the index of the position (r,c) in SNAKE_ORDER
        idx = SNAKE_ORDER.index((r,c)) if (r,c) in SNAKE_ORDER else 999
        # prefer idx small (close to corner). penalize if idx large
        if idx > 3:
            penalty -= 1600.0  # strong penalty for big tile away from corner area
    return penalty

def merge_clustering_reward(a: np.ndarray) -> float:
    """
    Reward large tiles being near their likely merge partner.
    E.g., reward proximity of 128 to 256, 64 to 128, etc.
    """
    reward = 0.0
    vals = sorted(set(a.flatten()) - {0})
    # for each pair (v -> 2v) if both present, reward proximity
    for v in vals:
        if v == 0: continue
        coords_v = tile_positions(a, v)
        coords_2v = tile_positions(a, v*2)
        if coords_v and coords_2v:
            # compute min distance
            dmin = min(manhattan(p,q) for p in coords_v for q in coords_2v)
            # reward closeness (closer = larger reward)
            reward += max(0, (6 - dmin)) * math.log2(v+1) * 60.0
    return reward

def smoothness_score(a: np.ndarray) -> float:
    logs = np.where(a>0, np.log2(a), 0.0)
    s = 0.0
    for r in range(4):
        for c in range(4):
            if a[r,c] == 0: continue
            v = logs[r,c]
            if c+1<4 and a[r,c+1]!=0:
                s -= abs(v - logs[r,c+1]) * 4.0
            if r+1<4 and a[r+1,c]!=0:
                s -= abs(v - logs[r+1,c]) * 4.0
    return s

def empties_score(a: np.ndarray) -> float:
    return float(np.sum(a==0)) * 200.0

def corner_lock_bonus(a: np.ndarray, corner=(3,3)) -> float:
    maxv = int(a.max())
    if maxv == 0:
        return 0.0
    r,c = corner
    if a[r,c] == maxv:
        return math.log2(maxv) * 300.0
    else:
        return -math.log2(maxv) * 800.0  # heavy penalty if max not in corner

def heuristic_combined(grid: List[List[int]], last_move: Optional[str]=None) -> float:
    """
    Combined heuristic tuned for strict snake enforcement:
    - heavy negative for any snake violations
    - heavy negative if large tiles not aligned near corner
    - reward clustering of merge partners
    - reward empties and smoothness moderately
    - corner lock strongly enforced
    """
    a = grid_to_array(grid)
    base = 0.0
    base += strict_tile_alignment_penalty(a)          # very large negative on violations
    base += merge_clustering_reward(a)                # encourages bringing pairs together
    base += smoothness_score(a)                       # smoother boards preferred
    base += empties_score(a)                          # survival
    base += corner_lock_bonus(a)                      # keep max tile in corner
    # small progress bonus
    maxv = int(a.max())
    base += math.log(maxv+1) * 48.0
    return float(base)

# Expectimax Agent 

class ExpectimaxAgent:

    def __init__(self, depth=3, preferred_corner=(3,3), skip_fours=True):
        self.depth = depth
        self.preferred_corner = preferred_corner
        self.skip_fours = skip_fours
        self.actions = ['up','down','left','right']
        self.cache = {}

    def reverse_move(self, mv: str) -> str:
        return {'up':'down','down':'up','left':'right','right':'left'}.get(mv, '')

    def breaks_corner(self, before_grid, after_grid):
        a = grid_to_array(before_grid)
        b = grid_to_array(after_grid)
        maxv = a.max()
        r,c = self.preferred_corner
        return (a[r,c] == maxv) and (b[r,c] != maxv)

    def increases_violations(self, before_grid, after_grid) -> bool:
        a = grid_to_array(before_grid)
        b = grid_to_array(after_grid)
        return snake_violations(b) > snake_violations(a)

    def get_best_move(self, game: Game2048) -> Optional[str]:
        empties = sum(row.count(0) for row in game.grid)
        depth = 4 if empties <= 3 else self.depth
        last_move = st.session_state.get("last_move", None)

        best_move = None
        best_val = -float('inf')
        violating = []

        current_max = max(max(row) for row in game.grid)

        # order actions by heuristic quick estimate
        for a in ['right','down','left','up']:
            g = game.clone()
            moved, _ = g.move(a)
            if not moved:
                continue

            # HARD corner: if move would move max tile out of corner, disallow unless it produces immediate significant merge
            immediate_merge = immediate_merge_max_value(g.grid)
            if self.breaks_corner(game.grid, g.grid) and not (immediate_merge >= current_max):
                # store as violating fallback
                violating.append((a,g))
                continue

            # also disallow moves that increase snake violations heavily
            if self.increases_violations(game.grid, g.grid):
                # allow as violating fallback only
                violating.append((a,g))
                continue

            # compute expectimax score
            val = self._expectimax(g, depth-1, is_player=False)

            # anti-oscillation: penalize immediate reverse of last move strongly
            if last_move and a == self.reverse_move(last_move):
                val -= 800.0

            # steer by heuristic combined
            val += heuristic_combined(g.grid, last_move)

            if val > best_val:
                best_val = val
                best_move = a

        # if no allowed move, choose best among violating (forced)
        if best_move is None and violating:
            best_v = -float('inf')
            for a,g in violating:
                v = heuristic_combined(g.grid, last_move)
                if v > best_v:
                    best_v = v
                    best_move = a

        # fallback to any legal move
        if best_move is None:
            for a in self.actions:
                g = game.clone()
                moved,_ = g.move(a)
                if not moved: continue
                v = self._expectimax(g, depth-1, is_player=False)
                if v > best_val:
                    best_val = v
                    best_move = a

        return best_move

    def _expectimax(self, game: Game2048, depth: int, is_player: bool):
        key = (tuple(tuple(r) for r in game.grid), depth, is_player)
        if key in self.cache:
            return self.cache[key]

        if depth == 0 or not game.can_move():
            val = heuristic_combined(game.grid, st.session_state.get("last_move", None))
            self.cache[key] = val
            return val

        if is_player:
            best = -float('inf')
            for mv in self.actions:
                g2 = game.clone()
                moved, _ = g2.move(mv)
                if not moved: continue
                val = self._expectimax(g2, depth-1, is_player=False)
                best = max(best, val)
            if best == -float('inf'):
                best = heuristic_combined(game.grid, st.session_state.get("last_move", None))
            self.cache[key] = best
            return best
        else:
            empties = [(r,c) for r in range(game.size) for c in range(game.size) if game.grid[r][c] == 0]
            if not empties:
                val = heuristic_combined(game.grid, st.session_state.get("last_move", None))
                self.cache[key] = val
                return val
            total = 0.0
            for (r,c) in empties:
                g2 = game.clone()
                g2.grid[r][c] = 2
                total += 0.9 * self._expectimax(g2, depth-1, is_player=True)
                if not self.skip_fours:
                    g4 = game.clone()
                    g4.grid[r][c] = 4
                    total += 0.1 * self._expectimax(g4, depth-1, is_player=True)
                else:
                    g4 = game.clone()
                    g4.grid[r][c] = 4
                    total += 0.1 * heuristic_combined(g4.grid, st.session_state.get("last_move", None))
            val = total / len(empties)
            self.cache[key] = val
            return val

# Streamlit UI

st.set_page_config(page_title="2048 AI — Strict Snake (Max enforcement)", page_icon="🎮", layout="centered")
st.title("2048 — Expectimax with MAX Snake Enforcement (A)")

# initialize session keys safely
if "game" not in st.session_state:
    st.session_state.game = Game2048()
if "agent" not in st.session_state:
    st.session_state.agent = ExpectimaxAgent(depth=3)
if "prev_grid" not in st.session_state:
    st.session_state.prev_grid = None
if "prev_prev_grid" not in st.session_state:
    st.session_state.prev_prev_grid = None
if "best_score" not in st.session_state:
    st.session_state.best_score = 0
if "autoplay_mode" not in st.session_state:
    st.session_state.autoplay_mode = "Off"
if "stagnant_count" not in st.session_state:
    st.session_state.stagnant_count = 0
if "last_move" not in st.session_state:
    st.session_state.last_move = None

game = st.session_state.game
agent = st.session_state.agent

# BUTTON HANDLERS 
btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([1,1,1,1])
btn_left  = btn_c1.button("⬅️ Left")
btn_up    = btn_c2.button("⬆️ Up")
btn_down  = btn_c3.button("⬇️ Down")
btn_right = btn_c4.button("➡️ Right")

ai_c1, ai_c2, ai_c3 = st.columns([1,1,2])
btn_ai = ai_c1.button("🤖 AI Move")
btn_restart = ai_c2.button("🔄 Restart")
depth = ai_c3.slider("Expectimax Base Depth", 1, 4, 3)
agent.depth = depth

autoplay_mode = st.selectbox(
    "Autoplay Mode",
    ["Off", "Play until no moves", "Play until target tile"],
    index=["Off","Play until no moves","Play until target tile"].index(st.session_state.autoplay_mode)
)
st.session_state.autoplay_mode = autoplay_mode
if autoplay_mode == "Play until target tile":
    target_tile = st.selectbox("Target Tile", [256,512,1024,2048], index=3)
    st.session_state.target_tile = target_tile

# Restart 
# --- BEFORE ANY UI IS RENDERED ---
if "restart" not in st.session_state:
    st.session_state.restart = False

if btn_restart:
    st.session_state.restart = True
    st.experimental_rerun()

if st.session_state.restart:
    # Reset only custom state variables (never internal Streamlit keys)
    for key in [
        "game", "agent", "prev_grid", "prev_prev_grid", "best_score",
        "autoplay_mode", "stagnant_count", "last_move", "target_tile"
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.restart = False
    st.session_state.game = Game2048()
    st.session_state.agent = ExpectimaxAgent(depth=3)
    st.experimental_rerun()


# perform manual moves and update last_move
if btn_left:
    moved,_ = game.move("left")
    if moved: st.session_state.last_move = "left"
if btn_right:
    moved,_ = game.move("right")
    if moved: st.session_state.last_move = "right"
if btn_up:
    moved,_ = game.move("up")
    if moved: st.session_state.last_move = "up"
if btn_down:
    moved,_ = game.move("down")
    if moved: st.session_state.last_move = "down"

# AI move
if btn_ai:
    mv = agent.get_best_move(game)
    if mv:
        moved,_ = game.move(mv)
        if moved:
            st.session_state.last_move = mv
        agent.cache.clear()  

# Rendering (after inputs)
st.markdown("""
<style>
.board { display:flex; flex-direction:column; align-items:center; margin-top:10px; }
.row { display:flex; gap:12px; margin-bottom:12px; }
.tile {
  width:86px; height:86px; border-radius:10px;
  display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:26px;
  transition: transform 0.12s ease, opacity 0.16s ease;
}
.tile.new { transform:scale(1.12); opacity:0; animation:appear 0.14s ease forwards; }
@keyframes appear { from {transform:scale(1.2); opacity:0} to {transform:scale(1); opacity:1} }
</style>
""", unsafe_allow_html=True)

COLORS = {
    0: "#efe6dd", 2:"#f7f2ec", 4:"#efe6d9", 8:"#f2b179", 16:"#f59563",
    32:"#f67c5f", 64:"#f65e3b", 128:"#eddc9a", 256:"#edcf74", 512:"#edc850",
    1024:"#edc53f", 2048:"#edc22e"
}

def board_html(grid, prev):
    tile_size = 86
    html = "<div class='board'>"
    for r in range(4):
        html += "<div class='row'>"
        for c in range(4):
            v = grid[r][c]
            col = COLORS.get(v, "#3c3a32")
            text = "#f9f6f2" if v >= 8 else "#776e65"
            is_new = prev is not None and prev[r][c] == 0 and v != 0
            new = " new" if is_new else ""
            label = str(v) if v != 0 else ""
            html += f"<div class='tile{new}' style='background:{col};color:{text};width:{tile_size}px;height:{tile_size}px'>{label}</div>"
        html += "</div>"
    html += "</div>"
    return html

html = board_html(game.grid, st.session_state.prev_grid)
st.markdown(html, unsafe_allow_html=True)
st.subheader(f"Score: {game.score}")

# update prev snapshots for animation
st.session_state.prev_prev_grid = copy.deepcopy(st.session_state.prev_grid) if st.session_state.prev_grid is not None else None
st.session_state.prev_grid = copy.deepcopy(game.grid)

# stagnation detection
if st.session_state.prev_prev_grid is not None:
    if st.session_state.prev_prev_grid == st.session_state.prev_grid:
        st.session_state.stagnant_count += 1
    else:
        st.session_state.stagnant_count = 0



# footer
if game.score > st.session_state.best_score:
    st.session_state.best_score = game.score
max_tile = max(max(row) for row in game.grid)
st.write(f"Max Tile: {max_tile} | Best Score: {st.session_state.best_score}")
st.markdown("<div style='font-size:13px;color:#666'>Mode: MAX Snake Enforcement (A). No debug overlay. Uploaded board image (for reference): /mnt/data/89f08ad2-5dbb-4ae5-83d1-b80cc1d80c5d.png</div>", unsafe_allow_html=True)







