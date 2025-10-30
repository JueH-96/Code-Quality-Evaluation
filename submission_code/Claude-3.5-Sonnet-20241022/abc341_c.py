H, W, N = map(int, input().split())
T = input()
grid = [input() for _ in range(H)]

# Get all possible starting positions (land cells)
possible_starts = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            possible_starts.append((i, j))

# For each move in T, store the direction
moves = []
for c in T:
    if c == 'L':
        moves.append((0, -1))
    elif c == 'R':
        moves.append((0, 1))
    elif c == 'U':
        moves.append((-1, 0))
    else:  # D
        moves.append((1, 0))

# For each possible starting position, try to follow the moves
# Store all valid ending positions
end_positions = set()

for start_i, start_j in possible_starts:
    # Try to follow moves from this starting position
    curr_i, curr_j = start_i, start_j
    valid = True
    
    # Store path of positions
    path = [(curr_i, curr_j)]
    
    # Follow each move
    for di, dj in moves:
        next_i = curr_i + di
        next_j = curr_j + dj
        
        # If we hit sea or go out of bounds, this starting position is invalid
        if (next_i < 0 or next_i >= H or next_j < 0 or next_j >= W or 
            grid[next_i][next_j] == '#'):
            valid = False
            break
            
        curr_i, curr_j = next_i, next_j
        path.append((curr_i, curr_j))
    
    # If path was valid, add ending position to set
    if valid:
        # Check if all positions in path were land
        if all(grid[i][j] == '.' for i, j in path):
            end_positions.add((curr_i, curr_j))

print(len(end_positions))