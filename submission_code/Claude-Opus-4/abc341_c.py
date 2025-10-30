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

# Reverse directions for backtracking
reverse_directions = {
    'L': 'R',
    'R': 'L',
    'U': 'D',
    'D': 'U'
}

valid_positions = 0

# Try each cell as a potential ending position
for end_row in range(H):
    for end_col in range(W):
        # Skip if ending position is sea
        if grid[end_row][end_col] == '#':
            continue
        
        # Backtrack from this position
        row, col = end_row, end_col
        valid = True
        
        # Go through the moves in reverse
        for i in range(N-1, -1, -1):
            # Get the reverse of the current move
            reverse_move = reverse_directions[T[i]]
            dr, dc = directions[reverse_move]
            
            # Move to previous position
            row += dr
            col += dc
            
            # Check if position is valid
            if row < 0 or row >= H or col < 0 or col >= W:
                valid = False
                break
            
            if grid[row][col] == '#':
                valid = False
                break
        
        if valid:
            valid_positions += 1

print(valid_positions)