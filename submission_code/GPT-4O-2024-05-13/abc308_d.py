# YOUR CODE HERE
from collections import deque

def is_valid_path(H, W, grid):
    target = "snuke"
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def is_valid(x, y, step):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == target[step % 5]
    
    queue = deque([(0, 0, 0)])  # (row, col, step)
    visited = set((0, 0))
    
    while queue:
        x, y, step = queue.popleft()
        
        if (x, y) == (H-1, W-1):
            return "Yes"
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, step + 1) and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, step + 1))
    
    return "No"

import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:]

print(is_valid_path(H, W, grid))