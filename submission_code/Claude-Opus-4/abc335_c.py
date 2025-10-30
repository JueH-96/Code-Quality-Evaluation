# YOUR CODE HERE
N, Q = map(int, input().split())

# Initialize positions of all parts
positions = [(i, 0) for i in range(N + 1)]  # 1-indexed, so we use N+1

# Process queries
for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        # Movement query
        direction = query[1]
        
        # Store current positions before moving
        old_positions = positions[:]
        
        # Move the head (part 1)
        x, y = positions[1]
        if direction == 'R':
            positions[1] = (x + 1, y)
        elif direction == 'L':
            positions[1] = (x - 1, y)
        elif direction == 'U':
            positions[1] = (x, y + 1)
        elif direction == 'D':
            positions[1] = (x, y - 1)
        
        # Each other part moves to where the previous part was
        for i in range(2, N + 1):
            positions[i] = old_positions[i - 1]
    
    else:
        # Position query
        p = int(query[1])
        x, y = positions[p]
        print(x, y)