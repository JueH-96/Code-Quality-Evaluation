from collections import deque

# Read input
N, M = map(int, input().split())
grid = []
for _ in range(N):
    grid.append(input().strip())

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# BFS to find all reachable positions
# A position is reachable if we can rest there (stop there)
reachable = set()
reachable.add((1, 1))  # Starting position (0-indexed)

# All ice squares we can touch (pass or rest)
touched = set()
touched.add((1, 1))

# BFS queue
queue = deque([(1, 1)])

while queue:
    curr_r, curr_c = queue.popleft()
    
    # Try moving in each direction
    for dr, dc in directions:
        # Keep moving in this direction until we hit a rock
        r, c = curr_r, curr_c
        
        # Track ice squares we pass through
        path = []
        
        while True:
            next_r = r + dr
            next_c = c + dc
            
            # Check if next position is within bounds
            if 0 <= next_r < N and 0 <= next_c < M:
                if grid[next_r][next_c] == '.':
                    # Ice square - keep moving
                    r, c = next_r, next_c
                    path.append((r, c))
                    touched.add((r, c))
                else:
                    # Rock - stop at current position
                    break
            else:
                # Out of bounds - stop
                break
        
        # If we moved to a new position, it's reachable
        if (r, c) != (curr_r, curr_c) and (r, c) not in reachable:
            reachable.add((r, c))
            queue.append((r, c))

# Count the number of touched ice squares
print(len(touched))