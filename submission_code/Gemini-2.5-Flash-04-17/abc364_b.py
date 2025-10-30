import sys

# Read H and W
h, w = map(int, sys.stdin.readline().split())

# Read starting position S_i, S_j (1-indexed)
start_r, start_c = map(int, sys.stdin.readline().split())

# Adjust to 0-indexed for internal use
current_r = start_r - 1
current_c = start_c - 1

# Read the grid
grid = []
for _ in range(h):
    grid.append(sys.stdin.readline().strip()) # strip() to remove trailing newline

# Read the sequence of moves
moves = sys.stdin.readline().strip()

# Process each move
for move in moves:
    # Determine the change in row and column based on the move
    dr, dc = 0, 0
    if move == 'U':
        dr = -1
    elif move == 'D':
        dr = 1
    elif move == 'L':
        dc = -1
    elif move == 'R':
        dc = 1

    # Calculate the potential next position
    potential_r = current_r + dr
    potential_c = current_c + dc

    # Check if the potential next cell is valid
    # It must be within bounds (0 <= r < h, 0 <= c < w)
    # AND it must be empty (grid[r][c] == '.')
    if 0 <= potential_r < h and 0 <= potential_c < w and grid[potential_r][potential_c] == '.':
        # If valid, update the current position
        current_r = potential_r
        current_c = potential_c
    # If not valid, the character stays at the current_r, current_c

# After all moves, print the final position (1-indexed)
print(current_r + 1, current_c + 1)