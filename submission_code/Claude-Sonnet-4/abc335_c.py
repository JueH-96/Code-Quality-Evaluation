# YOUR CODE HERE
N, Q = map(int, input().split())

# Initialize dragon positions
positions = []
for i in range(1, N + 1):
    positions.append((i, 0))

# Direction mappings
directions = {
    'R': (1, 0),
    'L': (-1, 0),
    'U': (0, 1),
    'D': (0, -1)
}

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        # Move head
        direction = query[1]
        dx, dy = directions[direction]
        
        # Store previous positions
        prev_positions = positions[:]
        
        # Move head
        head_x, head_y = positions[0]
        positions[0] = (head_x + dx, head_y + dy)
        
        # Each other part moves to where the part in front was
        for i in range(1, N):
            positions[i] = prev_positions[i - 1]
    
    else:  # query[0] == '2'
        # Report position
        p = int(query[1])
        x, y = positions[p - 1]  # Convert to 0-indexed
        print(x, y)