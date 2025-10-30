N, Q = map(int, input().split())

# Store positions of each part
positions = [(i,0) for i in range(1,N+1)]

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        # Movement query
        direction = query[1]
        
        # Store previous head position
        old_positions = positions.copy()
        
        # Move head
        x, y = positions[0]
        if direction == 'R':
            positions[0] = (x+1, y)
        elif direction == 'L':
            positions[0] = (x-1, y)
        elif direction == 'U':
            positions[0] = (x, y+1)
        else: # D
            positions[0] = (x, y-1)
            
        # Move other parts
        for i in range(1, N):
            positions[i] = old_positions[i-1]
            
    else:
        # Position query
        p = int(query[1])
        x, y = positions[p-1]
        print(x, y)