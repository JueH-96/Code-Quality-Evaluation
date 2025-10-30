# YOUR CODE HERE
from collections import deque

def bfs(grid, start, H, W):
    queue = deque([start])
    visited = set([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    
    while queue:
        x, y = queue.popleft()
        count += 1
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited and grid[nx][ny] == '.':
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return count

def max_degree_of_freedom(H, W, grid):
    max_freedom = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                # Check if any adjacent cell has a magnet
                if (i > 0 and grid[i-1][j] == '#') or (i < H-1 and grid[i+1][j] == '#') or \
                   (j > 0 and grid[i][j-1] == '#') or (j < W-1 and grid[i][j+1] == '#'):
                    continue
                # Perform BFS to find the degree of freedom
                freedom = bfs(grid, (i, j), H, W)
                max_freedom = max(max_freedom, freedom)
    
    return max_freedom

# Read input
import sys
input = sys.stdin.read
data = input().split()
H = int(data[0])
W = int(data[1])
grid = data[2:]

# Find and print the maximum degree of freedom
print(max_degree_of_freedom(H, W, grid))