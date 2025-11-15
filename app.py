# app.py — single-file 2048 + Expectimax + Streamlit UI
import streamlit as st
import random
import copy
import math
from typing import List, Tuple
import numpy as np

# ----------------- 2048 ENGINE (correct move logic) -----------------
class Game2048:
    def __init__(self, size=4, seed=None):
        self.size = size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    def reset(self):
        # start with two tiles, both forced to 2
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
        if force_two:
            self.grid[r][c] = 2
        else:
            self.grid[r][c] = 4 if random.random() < 0.1 else 2
        return True

    def can_move(self):
        # empty cell?
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == 0:
                    return True
        # horizontal merge possible?
        for r in range(self.size):
            for c in range(self.size-1):
                if self.grid[r][c] == self.grid[r][c+1]:
                    return True
        # vertical merge possible?
        for c in range(self.size):
            for r in range(self.size-1):
                if self.grid[r][c] == self.grid[r+1][c]:
                    return True
        return False

    @staticmethod
    def _move_row_left(row: List[int]) -> Tuple[List[int], int]:
        size = len(row)
        tight = [v for v in row if v != 0]
        new_row = []
        points = 0
        i = 0
        while i < len(tight):
            if i+1 < len(tight) and tight[i] == tight[i+1]:
                merged = tight[i]*2
                new_row.append(merged)
                points += merged
                i += 2
            else:
                new_row.append(tight[i])
                i += 1
        new_row += [0]*(size - len(new_row))
        return new_row, points

    def move(self, direction: str) -> Tuple[bool,int]:
        moved = False
        points_total = 0
        size = self.size

        if direction == 'left' or direction == 'right':
            for r in range(size):
                row = list(self.grid[r])
                if direction == 'right':
                    row = list(reversed(row))
                new_row, pts = self._move_row_left(row)
                if direction == 'right':
                    new_row = list(reversed(new_row))
                if new_row != self.grid[r]:
                    moved = True
                    self.grid[r] = new_row
                    points_total += pts
        elif direction == 'up' or direction == 'down':
            for c in range(size):
                col = [self.grid[r][c] for r in range(size)]
                if direction == 'down':
                    col = list(reversed(col))
                new_col, pts = self._move_row_left(col)
                if direction == 'down':
                    new_col = list(reversed(new_col))
                for r in range(size):
                    if self.grid[r][c] != new_col[r]:
                        moved = True
                    self.grid[r][c] = new_col[r]
                points_total += pts
        else:
            raise ValueError("Invalid direction")

        if moved:
            self.score += points_total
            self._add_random_tile()
        return moved, points_total

# ----------------- Heuristic & Expectimax Agent -----------------
def heuristic(grid: List[List[int]]) -> float:
    arr = np.array(grid)
    empties = float(np.sum(arr==0))
    max_tile = float(np.max(arr))
    smoothness = -calc_smoothness(arr)
    monotonicity = calc_monotonicity(arr)
    return 2.7*empties + 1.0*math.log(max_tile+1) + 0.1*smoothness + 1.0*monotonicity

def calc_smoothness(arr):
    s = 0.0
    rows,cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            if arr[r,c]==0: continue
            val = math.log(arr[r,c], 2)
            for (dr,dc) in [(0,1),(1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and arr[nr,nc]!=0:
                    nval = math.log(arr[nr,nc],2)
                    s += abs(val - nval)
    return s

def calc_monotonicity(arr):
    score = 0.0
    rows,cols = arr.shape
    for r in range(rows):
        vals = [0 if v==0 else math.log(v,2) for v in arr[r]]
        dec = sum(max(0, vals[i]-vals[i+1]) for i in range(len(vals)-1))
        inc = sum(max(0, vals[i+1]-vals[i]) for i in range(len(vals)-1))
        score += max(dec, inc)
    for c in range(cols):
        vals = [0 if v==0 else math.log(v,2) for v in arr[:,c]]
        dec = sum(max(0, vals[i]-vals[i+1]) for i in range(len(vals)-1))
        inc = sum(max(0, vals[i+1]-vals[i]) for i in range(len(vals)-1))
        score += max(dec, inc)
    return score

class ExpectimaxAgent:
    def __init__(self, depth=3):
        self.depth = depth
        self.actions = ['up','down','left','right']

    def get_best_move(self, game: Game2048) -> str:
        best_move = None
        best_score = -float('inf')
        for a in self.actions:
            g = game.clone()
            moved, _ = g.move(a)
            if not moved: continue
            val = self._expectimax(g, self.depth-1, is_player=False)
            if val > best_score:
                best_score = val
                best_move = a
        return best_move

    def _expectimax(self, game: Game2048, depth: int, is_player: bool):
        if depth == 0 or not game.can_move():
            return heuristic(game.grid)
        if is_player:
            best = -float('inf')
            for a in self.actions:
                g2 = game.clone()
                moved, _ = g2.move(a)
                if not moved: continue
                val = self._expectimax(g2, depth-1, is_player=False)
                if val > best: best = val
            return best if best!=-float('inf') else heuristic(game.grid)
        else:
            empties = [(r,c) for r in range(game.size) for c in range(game.size) if game.grid[r][c]==0]
            if not empties:
                return heuristic(game.grid)
            total = 0.0
            for (r,c) in empties:
                for tile_val, prob in [(2,0.9),(4,0.1)]:
                    g2 = game.clone()
                    g2.grid[r][c] = tile_val
                    total += prob * self._expectimax(g2, depth-1, is_player=True)
            return total / len(empties)

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="2048 AI", page_icon="🎮", layout="centered")
st.title("2048 — Play or Let the AI Play (Fixed)")

# session state — initialize once
if "game" not in st.session_state:
    st.session_state.game = Game2048()
if "agent" not in st.session_state:
    st.session_state.agent = ExpectimaxAgent(depth=3)
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False

game = st.session_state.game
agent = st.session_state.agent

# ---------- CSS + Animation ----------
st.markdown("""
<style>
.board { display:flex; flex-direction:column; align-items:center; }
.row { display:flex; gap:12px; margin-bottom:12px; }
.tile {
  width:94px; height:94px; border-radius:10px;
  display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:30px;
  transition: transform 0.14s ease, opacity 0.18s ease, box-shadow 0.12s ease;
  box-shadow: inset 0 -6px rgba(0,0,0,0.08);
}
.tile.new { transform: scale(1.12); opacity:0; animation: appear 160ms ease forwards; }
@keyframes appear {
  from { transform: scale(1.18); opacity:0; }
  to   { transform: scale(1); opacity:1; }
}
.container { background: transparent; padding: 6px; }
.controls { display:flex; gap:12px; margin-top:14px; justify-content:center; }
</style>
""", unsafe_allow_html=True)

# ---------- COLORS ----------
COLORS = {
    0: "#cdc1b4",
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e"
}

# ---------- BUTTONS FIRST (fix double-press) ----------
# Place buttons above render so clicks take effect this run
col_left, col_right, col_up, col_down = st.columns(4)
btn_left = col_left.button("⬅️ Left")
btn_right = col_right.button("➡️ Right")
btn_up = col_up.button("⬆️ Up")
btn_down = col_down.button("⬇️ Down")

# AI buttons
ai_c1, ai_c2, ai_c3 = st.columns([1,1,2])
btn_ai = ai_c1.button("🤖 AI Move")
btn_restart = ai_c2.button("🔄 Restart")
depth = ai_c3.slider("AI depth", 1, 4, 3)
st.session_state.agent = ExpectimaxAgent(depth=depth)
autoplay_toggle = st.checkbox("Autoplay (AI plays)", value=st.session_state.autoplay)
st.session_state.autoplay = autoplay_toggle

# Execute button actions (each click triggers a single move immediately)
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
    st.session_state.game = Game2048()
    st.experimental_rerun()

# Autoplay (no infinite Python loop). Do one AI step per render and use JS reload to continue.
if st.session_state.autoplay:
    if game.can_move():
        mv = agent.get_best_move(game)
        if mv:
            game.move(mv)
            # small JS reload to emulate continuous play; safe and avoids experimental_rerun loops
            st.markdown("<script>setTimeout(()=>window.location.reload(), 160);</script>", unsafe_allow_html=True)
    else:
        st.session_state.autoplay = False
        st.success("Autoplay finished — no moves left.")

# ---------- Render Board AS ONE HTML CHUNK (prevents stacking) ----------
def board_html_with_new_flags(grid, prev_grid):
    # We mark new tiles where prev_grid had 0 and grid has non-zero
    tile_size = 94
    gap = 12
    board_width = tile_size * len(grid[0]) + gap * (len(grid[0]) - 1)
    html = f"<div class='board'><div class='container' style='width:{board_width}px;'>"
    for r in range(len(grid)):
        html += "<div class='row'>"
        for c in range(len(grid[r])):
            v = grid[r][c]
            color = COLORS.get(v, "#3c3a32")
            label = str(v) if v != 0 else ""
            text_col = "#f9f6f2" if v >= 8 else "#776e65"
            # detect newly spawned tile for animation
            is_new = prev_grid is not None and prev_grid[r][c] == 0 and v != 0
            new_class = " new" if is_new else ""
            html += (f"<div class='tile{new_class}' style='background:{color};color:{text_col};"
                     f"width:{tile_size}px;height:{tile_size}px'>{label}</div>")
        html += "</div>"
    html += "</div></div>"
    return html

# For animation detection: we store previous grid in session_state
if "prev_grid" not in st.session_state:
    st.session_state.prev_grid = None

# Render board (after handling buttons)
html = board_html_with_new_flags(game.grid, st.session_state.prev_grid)
st.markdown(html, unsafe_allow_html=True)
st.subheader(f"Score: {game.score}")

# Update previous grid snapshot (deep copy)
st.session_state.prev_grid = copy.deepcopy(game.grid)

# ---------- Extra info ----------
max_tile = max(max(row) for row in game.grid)
st.write("Max tile:", max_tile)
