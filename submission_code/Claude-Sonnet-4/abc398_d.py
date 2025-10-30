# YOUR CODE HERE
N, R, C = map(int, input().split())
S = input().strip()

# Direction mappings
directions = {
    'N': (-1, 0),
    'S': (1, 0),
    'W': (0, -1),
    'E': (0, 1)
}

# Set to store all positions with smoke
smoke_positions = {(0, 0)}
result = []

for t in range(N):
    # Get wind direction
    wind = S[t]
    dr, dc = directions[wind]
    
    # Move all smoke
    new_positions = set()
    for r, c in smoke_positions:
        new_positions.add((r + dr, c + dc))
    
    smoke_positions = new_positions
    
    # Generate new smoke at (0,0) if needed
    if (0, 0) not in smoke_positions:
        smoke_positions.add((0, 0))
    
    # Check if smoke exists at target position
    if (R, C) in smoke_positions:
        result.append('1')
    else:
        result.append('0')

print(''.join(result))