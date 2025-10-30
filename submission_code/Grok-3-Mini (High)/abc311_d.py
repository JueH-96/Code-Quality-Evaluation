import sys
import collections

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
grid = []
for _ in range(N):
    row_str = data[index]
    index += 1
    grid.append(row_str)

# Define get_cell function to handle out of bounds
def get_cell(i, j):
    if i < 0 or i >= N or j < 0 or j >= M:
        return '#'
    return grid[i][j]

# Define find_end functions for each direction
def find_end_right(i, j):
    row = i
    col = j
    while True:
        if get_cell(row, col + 1) == '#':
            return (row, col)
        else:
            col += 1

def find_end_left(i, j):
    row = i
    col = j
    while True:
        if get_cell(row, col - 1) == '#':
            return (row, col)
        else:
            col -= 1

def find_end_down(i, j):
    row = i
    col = j
    while True:
        if get_cell(row + 1, col) == '#':
            return (row, col)
        else:
            row += 1

def find_end_up(i, j):
    row = i
    col = j
    while True:
        if get_cell(row - 1, col) == '#':
            return (row, col)
        else:
            row -= 1

# BFS to find all reachable stopping cells
queue = collections.deque()
visited = set()
start = (1, 1)  # 0-based indexing, start at (1,1)
queue.append(start)
visited.add(start)

while queue:
    curr = queue.popleft()
    # Compute neighbors for four directions
    neighbors = [
        find_end_right(curr[0], curr[1]),
        find_end_left(curr[0], curr[1]),
        find_end_down(curr[0], curr[1]),
        find_end_up(curr[0], curr[1])
    ]
    for nei in neighbors:
        if nei not in visited:
            visited.add(nei)
            queue.append(nei)

# R is the set of reachable stopping cells
R = visited

# Now create touched set, start with all reachable stopping cells
touched = set(R)

# Add cells occupied during slides for all possible moves between reachable cells
for S in R:
    # For right direction
    E_right = find_end_right(S[0], S[1])
    if E_right in R and E_right != S:
        row_s, col_s = S
        row_e, col_e = E_right  # same row, col_e >= col_s
        for c in range(col_s + 1, col_e + 1):  # cols from col_s+1 to col_e inclusive
            touched.add((row_s, c))
    
    # For left direction
    E_left = find_end_left(S[0], S[1])
    if E_left in R and E_left != S:
        row_s, col_s = S
        row_e, col_e = E_left  # same row, col_e <= col_s
        for c in range(col_e, col_s):  # cols from col_e to col_s-1 inclusive
            touched.add((row_s, c))
    
    # For down direction
    E_down = find_end_down(S[0], S[1])
    if E_down in R and E_down != S:
        row_s, col_s = S
        row_e, col_e = E_down  # same col, row_e >= row_s
        for r in range(row_s + 1, row_e + 1):  # rows from row_s+1 to row_e inclusive
            touched.add((r, col_s))
    
    # For up direction
    E_up = find_end_up(S[0], S[1])
    if E_up in R and E_up != S:
        row_s, col_s = S
        row_e, col_e = E_up  # same col, row_e <= row_s
        for r in range(row_e, row_s):  # rows from row_e to row_s-1 inclusive
            touched.add((r, col_s))

# The number of touched cells is the size of touched set
print(len(touched))