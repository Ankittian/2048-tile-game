# app.py — 2048 Game + Aggressive Merge-First Expectimax AI
# Manual controls fixed: one click = slide -> merge -> slide -> spawn
import streamlit as st
import random
import copy
import math
import numpy as np
from typing import List, Tuple, Optional

# ---------------------------
# 2048 Game Engine 
# ---------------------------
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
        Official 2048 movement logic (atomic):
        1) compress (slide)
        2) merge adjacent equal tiles (only once per move)
        3) compress again
        After all rows/cols, if any change occurred -> spawn new tile (2/4)
        """
        moved = False
        total_points = 0
        N = self.size

        def compress(row):
            new_row = [v for v in row if v != 0]
            new_row += [0] * (N - len(new_row))
            return new_row

        def merge(row):
            pts = 0
            for i in range(N - 1):
                if row[i] != 0 and row[i] == row[i+1]:
                    row[i] *= 2
                    pts += row[i]
                    row[i+1] = 0
            return row, pts

        for i in range(N):
            if direction in ('left', 'right'):
                row = self.grid[i][:]
                if direction == 'right':
                    row.reverse()

                row = compress(row)
                row, pts = merge(row)
                row = compress(row)

                if direction == 'right':
                    row.reverse()

                if row != self.grid[i]:
                    moved = True
                    self.grid[i] = row
                    total_points += pts

            else:  # up or down
                col = [self.grid[r][i] for r in range(N)]
                if direction == 'down':
                    col.reverse()

                col = compress(col)
                col, pts = merge(col)
                col = compress(col)

                if direction == 'down':
                    col.reverse()

                if col != [self.grid[r][i] for r in range(N)]:
                    moved = True
                    for r in range(N):
                        self.grid[r][i] = col[r]
                    total_points += pts

        if moved:
            self.score += total_points
            self._add_random_tile()

        return moved, total_points

# ---------------------------
# Heuristics
# ---------------------------
def count_potential_merges(a: np.ndarray) -> int:
    merges = 0
    for r in range(4):
        for c in range(3):
            if a[r,c] != 0 and a[r,c] == a[r,c+1]:
                merges += 1
    for c in range(4):
        for r in range(3):
            if a[r,c] != 0 and a[r,c] == a[r+1,c]:
                merges += 1
    return merges

def smoothness(a: np.ndarray) -> float:
    logs = np.where(a>0, np.log2(a), 0.0)
    s = 0.0
    for r in range(4):
        for c in range(4):
            if a[r,c] == 0: continue
            v = logs[r,c]
            if c+1<4 and a[r,c+1]!=0:
                s -= abs(v - logs[r,c+1])
            if r+1<4 and a[r+1,c]!=0:
                s -= abs(v - logs[r+1,c])
    return s

def rank_weighted_monotonicity(a: np.ndarray) -> float:
    logs = np.where(a>0, np.log2(a), 0.0)
    score = 0.0
    for r in range(4):
        for c in range(3):
            if logs[r,c] < logs[r,c+1]:
                score -= (logs[r,c] + logs[r,c+1]) * 0.5
    for c in range(4):
        for r in range(3):
            if logs[r,c] < logs[r+1,c]:
                score -= (logs[r,c] + logs[r+1,c]) * 0.5
    return score

def corner_row_integrity(a: np.ndarray, corner=(3,3)) -> float:
    logs = np.where(a>0, np.log2(a), 0.0)
    r, c = corner
    val = 0.0
    val += np.sum(np.diff(logs[r,:]) <= 0) * 5.0
    val += np.sum(np.diff(logs[:,c]) <= 0) * 5.0
    return val

def snake_score(a: np.ndarray) -> float:
    order = [
        (3,0),(3,1),(3,2),(3,3),
        (2,3),(2,2),(2,1),(2,0),
        (1,0),(1,1),(1,2),(1,3),
        (0,3),(0,2),(0,1),(0,0)
    ]
    logs = np.where(a>0, np.log2(a), 0.0)
    score = 0.0
    for i in range(len(order)-1):
        r1,c1 = order[i]
        r2,c2 = order[i+1]
        v1 = logs[r1,c1]
        v2 = logs[r2,c2]
        if v1 >= v2:
            score += v1
        else:
            score -= 3.0 * v1
    return score

def immediate_merge_value(grid: List[List[int]]) -> int:
    a = np.array(grid)
    val = 0
    for r in range(4):
        for c in range(3):
            if a[r,c] != 0 and a[r,c] == a[r,c+1]:
                val += a[r,c]*2
    for c in range(4):
        for r in range(3):
            if a[r,c] != 0 and a[r,c] == a[r+1,c]:
                val += a[r,c]*2
    return val

def max_tile_merge_pressure(a: np.ndarray) -> float:
    reward = 0.0
    for r in range(4):
        for c in range(3):
            if a[r,c] != 0 and a[r,c] == a[r,c+1]:
                v = a[r,c] * 2
                reward += (math.log2(v) ** 3) * 40.0
    for c in range(4):
        for r in range(3):
            if a[r,c] != 0 and a[r,c] == a[r+1,c]:
                v = a[r,c] * 2
                reward += (math.log2(v) ** 3) * 40.0
    return reward

def merge_toward_corner_bonus(grid: List[List[int]], corner=(3,3)) -> float:
    a = np.array(grid)
    cr, cc = corner
    bonus = 0.0
    for r in range(4):
        for c in range(3):
            if a[r,c] != 0 and a[r,c] == a[r,c+1]:
                if cc == 3:
                    bonus += math.log2(a[r,c]) * 90.0
    for c in range(4):
        for r in range(3):
            if a[r,c] != 0 and a[r,c] == a[r+1,c]:
                if cr == 3:
                    bonus += math.log2(a[r,c]) * 90.0
    return bonus

def immediate_merge_max_value(grid: List[List[int]]) -> int:
    a = np.array(grid)
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

# ---------------------------
# Expectimax Agent 
# ---------------------------
class ExpectimaxAgent:
    def __init__(self, depth=3, preferred_corner=(3,3), skip_fours=True):
        self.depth = depth
        self.preferred_corner = preferred_corner
        self.skip_fours = skip_fours
        self.actions = ['up','down','left','right']
        self.cache = {}

    def dynamic_priority(self, game: Game2048) -> List[str]:
        empties = sum(row.count(0) for row in game.grid)
        if empties >= 7:
            return ['down','right','left','up']
        if 3 <= empties < 7:
            return ['down','right','up','left']
        return ['right','down','up','left']

    def breaks_corner(self, before_grid, after_grid):
        a = np.array(before_grid)
        b = np.array(after_grid)
        maxv = a.max()
        r,c = self.preferred_corner
        return (a[r,c] == maxv) and (b[r,c] != maxv)

    def get_best_move(self, game: Game2048) -> Optional[str]:
        empties = sum(row.count(0) for row in game.grid)
        depth = 4 if empties <= 3 else self.depth
        preferred_order = self.dynamic_priority(game)

        best_move = None
        best_val = -float('inf')
        violating = []

        current_max = max(max(row) for row in game.grid)

        for a in preferred_order:
            g = game.clone()
            moved, _ = g.move(a)
            if not moved:
                continue

            immediate_best_merge = immediate_merge_max_value(g.grid)
            aggressive_merge_allowed = immediate_best_merge >= current_max

            if self.breaks_corner(game.grid, g.grid) and not aggressive_merge_allowed:
                violating.append((a,g))
                continue

            val = self._expectimax(g, depth-1, is_player=False)

            val += max_tile_merge_pressure(np.array(g.grid))
            val += merge_toward_corner_bonus(g.grid, self.preferred_corner)
            future_adj = immediate_merge_max_value(g.grid)
            if future_adj > 0:
                val += math.log1p(future_adj) * 160.0

            val += snake_score(np.array(g.grid)) * 3.0

            empties_before = sum(row.count(0) for row in game.grid)
            empties_after = sum(row.count(0) for row in g.grid)
            val += (empties_after - empties_before) * 250.0

            if val > best_val:
                best_val = val
                best_move = a

        if best_move is None and violating:
            best_v = -float('inf')
            for a,g in violating:
                val = heuristic_combined(g.grid, corner=self.preferred_corner)
                val += max_tile_merge_pressure(np.array(g.grid))
                val += merge_toward_corner_bonus(g.grid, self.preferred_corner)
                if val > best_v:
                    best_v = val
                    best_move = a

        if best_move is None:
            for a in self.actions:
                g = game.clone()
                moved, _ = g.move(a)
                if not moved: continue
                val = self._expectimax(g, depth-1, is_player=False)
                if val > best_val:
                    best_val = val
                    best_move = a

        return best_move

    def _expectimax(self, game: Game2048, depth: int, is_player: bool):
        key = (tuple(tuple(r) for r in game.grid), depth, is_player)
        if key in self.cache:
            return self.cache[key]

        if depth == 0 or not game.can_move():
            val = heuristic_combined(game.grid, corner=self.preferred_corner)
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
                best = heuristic_combined(game.grid, corner=self.preferred_corner)
            self.cache[key] = best
            return best

        else:
            empties = [(r,c) for r in range(game.size) for c in range(game.size) if game.grid[r][c] == 0]
            if not empties:
                val = heuristic_combined(game.grid, corner=self.preferred_corner)
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
                    total += 0.1 * heuristic_combined(g4.grid, corner=self.preferred_corner)
            val = total / len(empties)
            self.cache[key] = val
            return val

# ---------------------------
# Full heuristic )
# ---------------------------
def heuristic_combined(grid: List[List[int]], corner=(3,3)) -> float:
    a = np.array(grid, dtype=int)
    empties = float(np.sum(a == 0))
    merges = float(count_potential_merges(a))
    smooth = float(smoothness(a))
    rank_mono = float(rank_weighted_monotonicity(a))
    corner_integrity = float(corner_row_integrity(a, corner))
    snake = float(snake_score(a))
    max_tile = int(a.max()) if a.max() > 0 else 0

    if max_tile > 0:
        if a[corner] == max_tile:
            corner_bonus = float(max_tile) * 8.0
        else:
            corner_bonus = -float(max_tile) * 6.0
    else:
        corner_bonus = 0.0

    merge_pressure = max_tile_merge_pressure(a)
    merge_dir_bonus = merge_toward_corner_bonus(grid, corner)

    score = 0.0
    score += empties * 240.0
    score += merges * 120.0
    score += smooth * 1.0
    score += rank_mono * 6.0
    score += corner_integrity * 16.0
    score += snake * 2.5
    score += corner_bonus
    score += merge_pressure
    score += merge_dir_bonus
    score += math.log(max_tile+1) * 36.0

    return float(score)

# ---------------------------
# Streamlit UI - 
# ---------------------------
st.set_page_config(page_title="2048 AI (Aggressive Merge)", page_icon="🎮", layout="centered")
st.title("2048 — Aggressive Merge-First Expectimax AI (Option A)")

# initialize session state
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

game = st.session_state.game
agent = st.session_state.agent

# Button Controls

btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1,1,1,1])
btn_left  = btn_col1.button("⬅️ Left")
btn_up    = btn_col2.button("⬆️ Up")
btn_down  = btn_col3.button("⬇️ Down")
btn_right = btn_col4.button("➡️ Right")

# AI controls
ai_c1, ai_c2, ai_c3 = st.columns([1,1,2])
btn_ai = ai_c1.button("🤖 AI Move")
btn_restart = ai_c2.button("🔄 Restart")
depth = ai_c3.slider("Expectimax Base Depth", 1, 5, 3)
agent.depth = depth


# Perform actions based on clicks — these happen BEFORE rendering the board
if btn_left:
    game.move("left")
if btn_right:
    game.move("right")
if btn_up:
    game.move("up")
if btn_down:
    game.move("down")

if btn_ai:
    mv = agent.get_best_move(game)
    if mv:
        game.move(mv)

if btn_restart:
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.session_state.game = Game2048()
    st.session_state.agent = ExpectimaxAgent(depth=3)
    st.session_state.prev_grid = None
    st.session_state.prev_prev_grid = None
    st.session_state.best_score = 0
    st.session_state.autoplay_mode = "Off"
    st.session_state.stagnant_count = 0

    st.rerun()  

# ---------------------------
# Board rendering 
# ---------------------------
# CSS + tile styles
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
    0: "#efe6dd",
    2: "#f7f2ec",
    4: "#efe6d9",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#eddc9a",
    256: "#edcf74",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e"
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

# Render board AFTER processing inputs
html = board_html(game.grid, st.session_state.prev_grid)
st.markdown(html, unsafe_allow_html=True)
st.subheader(f"Score: {game.score}")

# update previous grid snapshots for animation/new-tile detection
st.session_state.prev_prev_grid = copy.deepcopy(st.session_state.prev_grid) if st.session_state.prev_grid is not None else None
st.session_state.prev_grid = copy.deepcopy(game.grid)

# Track stagnation: if grid unchanged for two runs -> increment stagnant_count
if st.session_state.prev_prev_grid is not None:
    if st.session_state.prev_prev_grid == st.session_state.prev_grid:
        st.session_state.stagnant_count += 1
    else:
        st.session_state.stagnant_count = 0




# Footer info
if game.score > st.session_state.best_score:
    st.session_state.best_score = game.score
max_tile = max(max(row) for row in game.grid)
st.write(f"Max Tile: {max_tile} | Best Score: {st.session_state.best_score}")
st.markdown("<div style='font-size:13px;color:#666'>Agent: Aggressive merge-first Expectimax. Manual controls are atomic (1-click = full move + spawn).</div>", unsafe_allow_html=True)




