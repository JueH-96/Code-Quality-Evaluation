from collections import deque

N, M = map(int, input().split())
grid = []
for _ in range(N):
    grid.append(input().strip())

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Set to store all reachable positions
reachable = set()

# Queue for BFS
queue = deque()

# Start position (2,2) in 1-indexed becomes (1,1) in 0-indexed
start = (1, 1)
queue.append(start)
reachable.add(start)

while queue:
    r, c = queue.popleft()
    
    # Try moving in each direction
    for dr, dc in directions:
        # Start from current position
        nr, nc = r, c
        path = []
        
        # Keep moving until hitting a rock
        while True:
            # Next position
            next_r, next_c = nr + dr, nc + dc
            
            # Check if next position is within bounds and is ice
            if (0 <= next_r < N and 0 <= next_c < M and 
                grid[next_r][next_c] == '.'):
                nr, nc = next_r, next_c
                path.append((nr, nc))
            else:
                # Hit a rock or boundary, stop here
                break
        
        # Add all positions in the path to reachable set
        for pos in path:
            if pos not in reachable:
                reachable.add(pos)
                queue.append(pos)

print(len(reachable))