# YOUR CODE HERE
def count_reachable_ice_squares(N, M, grid):
    from collections import deque

    # Directions for movement: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Starting position
    start = (2, 2)
    
    # Queue for BFS
    queue = deque([start])
    
    # Set to keep track of visited squares
    visited = set([start])
    
    # BFS to explore all reachable ice squares
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = x, y
            # Move in the direction until hitting a rock
            while grid[nx][ny] == '.':
                nx += dx
                ny += dy
            # Step back to the last ice square
            nx -= dx
            ny -= dy
            # If this square has not been visited, add it to the queue
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return len(visited)

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
grid = data[2:]

# Calculate and print the result
result = count_reachable_ice_squares(N, M, grid)
print(result)