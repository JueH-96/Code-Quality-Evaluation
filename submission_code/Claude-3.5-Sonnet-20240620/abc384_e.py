# YOUR CODE HERE
from collections import deque

def can_absorb(current_strength, target_strength, X):
    return target_strength < current_strength / X

def bfs(grid, H, W, X, P, Q):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = set()
    queue = deque([(P-1, Q-1)])
    total_strength = grid[P-1][Q-1]

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if can_absorb(total_strength, grid[nr][nc], X):
                    total_strength += grid[nr][nc]
                    queue.append((nr, nc))

    return total_strength

# Read input
H, W, X = map(int, input().split())
P, Q = map(int, input().split())
grid = [list(map(int, input().split())) for _ in range(H)]

# Solve and print result
result = bfs(grid, H, W, X, P, Q)
print(result)