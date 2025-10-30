# YOUR CODE HERE
from collections import deque

def can_reach_end(H, W, grid):
    target = "snuke"
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    queue = deque([(0, 0, 0)])
    visited = set([(0, 0, 0)])
    
    while queue:
        i, j, idx = queue.popleft()
        
        if i == H - 1 and j == W - 1:
            return "Yes"
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            next_idx = (idx + 1) % 5
            
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == target[next_idx]:
                if (ni, nj, next_idx) not in visited:
                    visited.add((ni, nj, next_idx))
                    queue.append((ni, nj, next_idx))
    
    return "No"

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

print(can_reach_end(H, W, grid))