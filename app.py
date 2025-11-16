# app.py — 2048 Game + Expectimax AI using SO reference heuristics
import streamlit as st
import random
import copy
import math
import numpy as np
from typing import List, Tuple, Optional

# ------------------------------------------------------------
#                    2048 GAME ENGINE
# ------------------------------------------------------------
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
        empties = [(r,c) for r in range(self.size) if True for c in range(self.size) if self.grid[r][c]==0]
        if not empties:
            return False
        r,c = random.choice(empties)
        self.grid[r][c] = 2 if force_two else (4 if random.random()<0.1 else 2)
        return True

    def can_move(self):
        # empty cell exists
        if any(self.grid[r][c]==0 for r in range(self.size) for c in range(self.size)):
            return True
        # horizontal merge
        for r in range(self.size):
            for c in range(self.size-1):
                if self.grid[r][c] == self.grid[r][c+1]:
                    return True
        # vertical merge
        for c in range(self.size):
            for r in range(self.size-1):
                if self.grid[r][c] == self.grid[r+1][c]:
                    return True
        return False

    @staticmethod
    def _move_row_left(row: List[int]):
        tight = [v for v in row if v!=0]
        new = []
        pts = 0
        i=0
        while i < len(tight):
            if i+1 < len(tight) and tight[i] == tight[i+1]:
                merged = tight[i]*2
                new.append(merged)
                pts += merged
                i += 2
            else:
                new.append(tight[i])
                i += 1
        new += [0]*(len(row)-len(new))
        return new, pts

    def move(self, direction: str):
        moved=False
        total_pts=0
        N=self.size

        if direction in ('left','right'):
            for r in range(N):
                row=list(self.grid[r])
                if direction=='right':
                    row=row[::-1]
                new_row, pts = self._move_row_left(row)
                if direction=='right':
                    new_row=new_row[::-1]
                if new_row != self.grid[r]:
                    moved=True
                    self.grid[r] = new_row
                    total_pts += pts

        elif direction in ('up','down'):
            for c in range(N):
                col=[self.grid[r][c] for r in range(N)]
                if direction=='down':
                    col=col[::-1]
                new_col, pts = self._move_row_left(col)
                if direction=='down':
                    new_col=new_col[::-1]
                for r in range(N):
                    if self.grid[r][c] != new_col[r]:
                        moved=True
                    self.grid[r][c] = new_col[r]
                total_pts += pts

        if moved:
            self.score += total_pts
            self._add_random_tile()
        return moved, total_pts

# ------------------------------------------------------------
#              HEURISTICS FROM STACKOVERFLOW
# ------------------------------------------------------------
def count_potential_merges(a):
    merges=0
    for r in range(4):
        for c in range(3):
            if a[r,c]!=0 and a[r,c]==a[r,c+1]:
                merges+=1
    for c in range(4):
        for r in range(3):
            if a[r,c]!=0 and a[r,c]==a[r+1,c]:
                merges+=1
    return merges

def smoothness(a):
    logs=np.where(a>0, np.log2(a), 0)
    s=0
    for r in range(4):
        for c in range(4):
            if a[r,c]==0: continue
            v=logs[r,c]
            if c+1<4 and a[r,c+1]!=0:
                s -= abs(v-logs[r,c+1])
            if r+1<4 and a[r+1,c]!=0:
                s -= abs(v-logs[r+1,c])
    return s

def rank_weighted_monotonicity(a):
    logs=np.where(a>0, np.log2(a),0)
    score=0
    # rows
    for r in range(4):
        for c in range(3):
            if logs[r,c] < logs[r,c+1]:
                score -= (logs[r,c]+logs[r,c+1])*0.5
    # cols
    for c in range(4):
        for r in range(3):
            if logs[r,c] < logs[r+1,c]:
                score -= (logs[r,c]+logs[r+1,c])*0.5
    return score

def corner_row_integrity(a, corner=(3,3)):
    logs=np.where(a>0, np.log2(a),0)
    r,c=corner
    val=0.0
    # check bottom row
    row=logs[r,:]
    val+= np.sum(np.diff(row)<=0)*5
    # right column
    col=logs[:,c]
    val+= np.sum(np.diff(col)<=0)*5
    return val

def heuristic(grid, corner=(3,3)):
    a=np.array(grid)
    empties=float(np.sum(a==0))
    merges=float(count_potential_merges(a))
    smooth=float(smoothness(a))
    mono=float(rank_weighted_monotonicity(a))
    corner_int=float(corner_row_integrity(a, corner))
    max_tile = int(a.max()) if a.max()>0 else 0

    if max_tile>0:
        if a[corner]==max_tile:
            corner_bonus = max_tile * 12
        else:
            corner_bonus = -max_tile * 24
    else:
        corner_bonus=0

    score = (
        empties*270 +
        merges*110 +
        smooth*1 +
        mono*8 +
        corner_int*18 +
        corner_bonus +
        math.log(max_tile+1)*40
    )
    return float(score)

# ------------------------------------------------------------
#                EXPECTIMAX AGENT
# ------------------------------------------------------------
class ExpectimaxAgent:
    def __init__(self, depth=3, preferred_corner=(3,3), skip_fours=True):
        self.depth=depth
        self.preferred_corner = preferred_corner
        self.skip_fours = skip_fours
        self.actions=['up','down','left','right']
        self.preferred=['down','right','left','up']
        self.cache={}

    def max_in_corner(self, grid):
        a=np.array(grid)
        return a[self.preferred_corner] == a.max()

    def breaks_corner(self, before_grid, after_grid):
        a=np.array(before_grid)
        b=np.array(after_grid)
        max_tile=a.max()
        r,c=self.preferred_corner
        if a[r,c] == max_tile and b[r,c] != max_tile:
            return True
        return False

    def get_best_move(self, game):
        empties=sum(row.count(0) for row in game.grid)
        depth=4 if empties<=3 else self.depth

        best_move=None
        best_score=-1e18
        violating=[]

        for a in self.preferred:
            g=game.clone()
            moved,_=g.move(a)
            if not moved: continue

            if self.breaks_corner(game.grid, g.grid):
                violating.append((a,g))
                continue

            val=self._expectimax(g, depth-1, False)
            if val > best_score:
                best_score=val
                best_move=a

        if best_move is None and violating:
            best_val=-1e18
            for a,g in violating:
                v=heuristic(g.grid, self.preferred_corner)
                if v > best_val:
                    best_val=v
                    best_move=a

        if best_move is None:
            for a in self.actions:
                g=game.clone()
                moved,_=g.move(a)
                if not moved: continue
                val=self._expectimax(g, depth-1, False)
                if val > best_score:
                    best_score=val
                    best_move=a

        return best_move

    def _expectimax(self, game, depth, is_player):
        key=(tuple(tuple(r) for r in game.grid), depth, is_player)
        if key in self.cache:
            return self.cache[key]

        if depth==0 or not game.can_move():
            val=heuristic(game.grid, self.preferred_corner)
            self.cache[key]=val
            return val

        if is_player:
            best=-1e18
            for a in self.actions:
                g=game.clone()
                moved,_=g.move(a)
                if not moved: continue
                val=self._expectimax(g, depth-1, False)
                best=max(best, val)
            if best==-1e18:
                best=heuristic(game.grid, self.preferred_corner)
            self.cache[key]=best
            return best

        else:
            empties=[(r,c) for r in range(4) for c in range(4) if game.grid[r][c]==0]
            if not empties:
                val=heuristic(game.grid, self.preferred_corner)
                self.cache[key]=val
                return val

            total=0
            for r,c in empties:
                g2=game.clone()
                g2.grid[r][c]=2
                total += 0.9*self._expectimax(g2, depth-1, True)

                if not self.skip_fours:
                    g4=game.clone()
                    g4.grid[r][c]=4
                    total += 0.1*self._expectimax(g4, depth-1, True)
                else:
                    g4=game.clone()
                    g4.grid[r][c]=4
                    total += 0.1*heuristic(g4.grid, self.preferred_corner)

            val=total/len(empties)
            self.cache[key]=val
            return val

# ------------------------------------------------------------
#                 STREAMLIT UI
# ------------------------------------------------------------
st.set_page_config(page_title="2048 AI", page_icon="🎮", layout="centered")
st.title("2048 — Expectimax AI (SO Strategies)")

if "game" not in st.session_state:
    st.session_state.game=Game2048()
if "agent" not in st.session_state:
    st.session_state.agent=ExpectimaxAgent(depth=3)
if "autoplay" not in st.session_state:
    st.session_state.autoplay=False
if "prev_grid" not in st.session_state:
    st.session_state.prev_grid=None
if "best_score" not in st.session_state:
    st.session_state.best_score=0

game=st.session_state.game
agent=st.session_state.agent

# UI CSS
st.markdown("""
<style>
.board { display:flex; flex-direction:column; align-items:center; }
.row { display:flex; gap:12px; margin-bottom:12px; }
.tile {
  width:86px; height:86px; border-radius:10px;
  display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:26px;
  transition: transform 0.12s ease, opacity 0.16s ease;
}
.tile.new { transform:scale(1.12); opacity:0; animation:appear 140ms ease forwards; }
@keyframes appear { from {transform:scale(1.18); opacity:0} to {transform:scale(1); opacity:1} }
</style>
""", unsafe_allow_html=True)

COLORS={
    0:"#efe6dd",
    2:"#f7f2ec",
    4:"#efe6d9",
    8:"#f2b179",
    16:"#f59563",
    32:"#f67c5f",
    64:"#f65e3b",
    128:"#eddc9a",
    256:"#edcf74",
    512:"#edc850",
    1024:"#edc53f",
    2048:"#edc22e"
}

def board_html(grid, prev):
    tile_size=86
    gap=12
    html="<div class='board'>"
    for r in range(4):
        html+="<div class='row'>"
        for c in range(4):
            v=grid[r][c]
            col=COLORS.get(v,"#3c3a32")
            text="#f9f6f2" if v>=8 else "#776e65"
            is_new=prev and prev[r][c]==0 and v!=0
            new_class=" new" if is_new else ""
            label=str(v) if v!=0 else ""
            html+=f"<div class='tile{new_class}' style='background:{col};color:{text};width:{tile_size}px;height:{tile_size}px'>{label}</div>"
        html+="</div>"
    html+="</div>"
    return html

# Render board FIRST
html = board_html(game.grid, st.session_state.prev_grid)
st.markdown(html, unsafe_allow_html=True)
st.subheader(f"Score: {game.score}")

st.session_state.prev_grid = copy.deepcopy(game.grid)

# ------------------------------------------------------------
#           CONTROLS BELOW THE BOARD
# ------------------------------------------------------------
st.markdown("### Controls")

c1,c2,c3,c4 = st.columns(4)
left_btn  = c1.button("⬅️ Left")
up_btn    = c2.button("⬆️ Up")
down_btn  = c3.button("⬇️ Down")
right_btn = c4.button("➡️ Right")

if left_btn:  game.move("left")
if right_btn: game.move("right")
if up_btn:    game.move("up")
if down_btn:  game.move("down")

# AI Controls
st.markdown("### AI Controls")
ai_c1, ai_c2, ai_c3 = st.columns([1,1,2])
ai_btn = ai_c1.button("🤖 AI Move")
restart_btn = ai_c2.button("🔄 Restart")
depth = ai_c3.slider("Search Depth", 1, 4, 3)
agent.depth = depth

autoplay = st.checkbox("Autoplay AI", value=st.session_state.autoplay)
st.session_state.autoplay = autoplay

if ai_btn:
    mv=agent.get_best_move(game)
    if mv:
        game.move(mv)

if restart_btn:
    st.session_state.game = Game2048()
    st.session_state.prev_grid = None
    st.session_state.agent.cache.clear()
    st.session_state.autoplay=False
    st.rerun()

# Autoplay
if st.session_state.autoplay:
    if game.can_move():
        mv=agent.get_best_move(game)
        if mv:
            game.move(mv)
            st.markdown("<script>setTimeout(()=>window.location.reload(),160);</script>", unsafe_allow_html=True)
    else:
        st.session_state.autoplay=False
        st.success("Autoplay finished.")

max_tile=max(max(row) for row in game.grid)
st.write(f"Max Tile: {max_tile}")



# ---------- Extra info ----------
max_tile = max(max(row) for row in game.grid)
st.write("Max tile:", max_tile)

