# YOUR CODE HERE
from collections import deque

H, W, X = map(int, input().split())
P, Q = map(int, input().split())
P -= 1  # Convert to 0-indexed
Q -= 1

grid = []
for _ in range(H):
    row = list(map(int, input().split()))
    grid.append(row)

# BFS to find maximum strength
max_strength = grid[P][Q]
visited = [[False] * W for _ in range(H)]
queue = deque([(P, Q, grid[P][Q])])
visited[P][Q] = True

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# We need to explore all possible paths, so we'll use a different approach
# We'll use a priority queue or explore all reachable cells
def can_reach(start_r, start_c, start_strength):
    # Find all cells reachable from (start_r, start_c) with given strength
    reachable = set()
    reachable.add((start_r, start_c))
    q = deque([(start_r, start_c, start_strength)])
    vis = [[False] * W for _ in range(H)]
    vis[start_r][start_c] = True
    
    while q:
        r, c, strength = q.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not vis[nr][nc]:
                if grid[nr][nc] * X < strength:
                    vis[nr][nc] = True
                    reachable.add((nr, nc))
                    q.append((nr, nc, strength + grid[nr][nc]))
    
    return reachable

# Try all possible orderings using DFS
def dfs(r, c, strength, absorbed):
    global max_strength
    max_strength = max(max_strength, strength)
    
    # Try to absorb adjacent slimes
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in absorbed:
            if grid[nr][nc] * X < strength:
                absorbed.add((nr, nc))
                dfs(nr, nc, strength + grid[nr][nc], absorbed)
                absorbed.remove((nr, nc))

absorbed = set()
absorbed.add((P, Q))
dfs(P, Q, grid[P][Q], absorbed)

print(max_strength)