# YOUR CODE HERE
N, Q = map(int, input().split())

# Initialize the positions of all parts
positions = [(i, 0) for i in range(N)]

# Define movement directions
directions = {
    'R': (1, 0),
    'L': (-1, 0),
    'U': (0, 1),
    'D': (0, -1)
}

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        # Move the dragon
        direction = query[1]
        dx, dy = directions[direction]
        
        # Store the current head position
        prev_x, prev_y = positions[0]
        
        # Move the head
        positions[0] = (prev_x + dx, prev_y + dy)
        
        # Move the rest of the body
        for i in range(1, N):
            positions[i], (prev_x, prev_y) = (prev_x, prev_y), positions[i]
    
    elif query[0] == '2':
        # Query the position of a part
        p = int(query[1]) - 1  # Convert to 0-based index
        print(f"{positions[p][0]} {positions[p][1]}")