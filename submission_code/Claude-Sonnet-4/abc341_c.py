# YOUR CODE HERE
H, W, N = map(int, input().split())
T = input().strip()
grid = []
for _ in range(H):
    grid.append(input().strip())

# Direction mappings
directions = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

def is_valid_position(r, c):
    return 0 <= r < H and 0 <= c < W and grid[r][c] == '.'

def simulate_path(start_r, start_c):
    r, c = start_r, start_c
    
    # Check if starting position is valid
    if not is_valid_position(r, c):
        return False
    
    # Follow each move in the sequence
    for move in T:
        dr, dc = directions[move]
        r += dr
        c += dc
        
        # Check if new position is valid
        if not is_valid_position(r, c):
            return False
    
    return True

# Count valid starting positions
count = 0
valid_endpoints = set()

# Try each land cell as a potential starting position
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.' and simulate_path(i, j):
            # Calculate final position
            final_r, final_c = i, j
            for move in T:
                dr, dc = directions[move]
                final_r += dr
                final_c += dc
            valid_endpoints.add((final_r, final_c))

print(len(valid_endpoints))