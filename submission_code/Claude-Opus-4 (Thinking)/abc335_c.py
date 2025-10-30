from collections import deque

n, q = map(int, input().split())

# Initialize positions
positions = deque([(i, 0) for i in range(1, n + 1)])

for _ in range(q):
    query = input().split()
    if query[0] == '1':
        # Move query
        direction = query[1]
        
        # Calculate the new head position
        x, y = positions[0]
        if direction == 'R':
            new_head = (x + 1, y)
        elif direction == 'L':
            new_head = (x - 1, y)
        elif direction == 'U':
            new_head = (x, y + 1)
        elif direction == 'D':
            new_head = (x, y - 1)
        
        # Add new head and remove tail
        positions.appendleft(new_head)
        positions.pop()
    else:
        # Position query
        p = int(query[1]) - 1  # Convert to 0-indexed
        x, y = positions[p]
        print(x, y)