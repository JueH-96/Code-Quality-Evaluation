def count_reachable_squares(grid, N, M):
    # Start position (0-indexed)
    start_r, start_c = 1, 1  # corresponds to position (2, 2) in 1-indexed
    
    # Sets to track the squares
    reachable = set([(start_r, start_c)])
    enqueued = set([(start_r, start_c)])
    
    # BFS queue
    queue = [(start_r, start_c)]
    
    # Four directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.pop(0)
        
        # Try all four directions
        for dr, dc in directions:
            nr, nc = r, c
            
            # Keep sliding in the chosen direction until hitting a rock
            while True:
                next_r, next_c = nr + dr, nc + dc
                
                if next_r < 0 or next_r >= N or next_c < 0 or next_c >= M or grid[next_r][next_c] == '#':
                    break
                
                nr, nc = next_r, next_c
                reachable.add((nr, nc))
            
            # If we've moved to a new position, add it to the queue for further exploration
            if (nr, nc) != (r, c) and (nr, nc) not in enqueued:
                queue.append((nr, nc))
                enqueued.add((nr, nc))
    
    return len(reachable)

# Read input
N, M = map(int, input().split())
grid = []
for _ in range(N):
    grid.append(input())

# Calculate and print the answer
print(count_reachable_squares(grid, N, M))