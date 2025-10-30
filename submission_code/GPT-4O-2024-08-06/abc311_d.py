def count_reachable_ice_squares(N, M, grid):
    from collections import deque
    
    # Directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Starting position
    start = (2, 2)
    
    # Visited set to keep track of visited ice squares
    visited = set()
    visited.add(start)
    
    # Queue for BFS
    queue = deque([start])
    
    # BFS to explore all reachable ice squares
    while queue:
        x, y = queue.popleft()
        
        # Explore all four directions
        for dx, dy in directions:
            nx, ny = x, y
            # Move in the direction until we hit a rock
            while True:
                nx += dx
                ny += dy
                if grid[nx][ny] == '#':
                    break
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    # The number of unique ice squares visited
    return len(visited)

# Reading input
import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
grid = data[1:]

# Calculate the result
result = count_reachable_ice_squares(N, M, grid)

# Output the result
print(result)