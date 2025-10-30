from collections import deque

def solve():
    N, M = map(int, input().split())
    grid = []
    for _ in range(N):
        grid.append(input().strip())
    
    # Convert 1-indexed start (2,2) to 0-indexed (1,1)
    start_row, start_col = 1, 1
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Set to store all ice squares that can be touched
    touched = set()
    touched.add((start_row, start_col))
    
    # Queue for BFS - stores positions where player can rest
    queue = deque([(start_row, start_col)])
    visited_rest = set([(start_row, start_col)])
    
    while queue:
        row, col = queue.popleft()
        
        # Try moving in each direction
        for dr, dc in directions:
            # Slide in this direction
            r, c = row, col
            
            # Keep sliding and record path
            while True:
                nr, nc = r + dr, c + dc
                
                # Check if we can move to next position
                if 0 <= nr < N and 0 <= nc < M and grid[nr][nc] == '.':
                    # Can move to this ice square
                    touched.add((nr, nc))
                    r, c = nr, nc
                else:
                    # Hit a rock or boundary
                    break
            
            # If we moved and haven't visited this rest position
            if (r, c) != (row, col) and (r, c) not in visited_rest:
                visited_rest.add((r, c))
                queue.append((r, c))
    
    print(len(touched))

solve()