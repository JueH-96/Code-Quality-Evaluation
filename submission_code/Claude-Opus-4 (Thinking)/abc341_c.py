# YOUR CODE HERE
H, W, N = map(int, input().split())
T = input().strip()
grid = []
for _ in range(H):
    grid.append(input().strip())

# Direction mappings
moves = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

# Set to store all possible current positions
possible_positions = set()

# Try each land cell as a starting position
for start_i in range(H):
    for start_j in range(W):
        # Skip if starting position is sea
        if grid[start_i][start_j] == '#':
            continue
        
        # Simulate the path
        valid = True
        i, j = start_i, start_j
        
        for move in T:
            di, dj = moves[move]
            i, j = i + di, j + dj
            
            # Check if out of bounds
            if i < 0 or i >= H or j < 0 or j >= W:
                valid = False
                break
            
            # Check if current cell is sea
            if grid[i][j] == '#':
                valid = False
                break
        
        # If the path is valid, add the ending position
        if valid:
            possible_positions.add((i, j))

print(len(possible_positions))