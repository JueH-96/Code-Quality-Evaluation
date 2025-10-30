import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read H, W, N
H = int(data[index])
W = int(data[index + 1])
N = int(data[index + 2])
index += 3

# Read the move string T
T = data[index]
index += 1

# Read the grid
grid = []
for i in range(H):
    row_str = data[index + i]
    grid.append(row_str)

# Define inverse direction dictionary
inv_dir_dict = {
    'L': 'R',
    'R': 'L',
    'U': 'D',
    'D': 'U'
}

# Define delta dictionary for moves
delta_dict = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

# Compute the inverse move sequence
T_rev = T[::-1]
inv_moves = [inv_dir_dict[c] for c in T_rev]

# Define function to check if the reverse path from a given end position is valid
def is_valid_path(r_end, c_end):
    current_r = r_end
    current_c = c_end
    for move in inv_moves:
        delta_r, delta_c = delta_dict[move]
        new_r = current_r + delta_r
        new_c = current_c + delta_c
        if not (0 <= new_r < H and 0 <= new_c < W and grid[new_r][new_c] == '.'):
            return False
        current_r, current_c = new_r, new_c
    return True

# Count the number of valid end positions
count = 0
for r in range(H):
    for c in range(W):
        if grid[r][c] == '.':  # Only consider land cells
            if is_valid_path(r, c):
                count += 1

# Output the result
print(count)