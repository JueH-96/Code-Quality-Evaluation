from collections import deque

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

def is_valid(r, c):
    return 0 <= r < H and 0 <= c < W

def can_move_from(r, c):
    # Check if we can move from (r, c) by checking if any adjacent cell has a magnet
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_valid(nr, nc) and grid[nr][nc] == '#':
            return False
    return True

def bfs(start_r, start_c):
    visited = set()
    queue = deque([(start_r, start_c)])
    visited.add((start_r, start_c))
    count = 0
    
    while queue:
        r, c = queue.popleft()
        count += 1
        
        # If we can move from this cell, explore adjacent empty cells
        if can_move_from(r, c):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and grid[nr][nc] == '.' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    return count

max_degree = 0
for r in range(H):
    for c in range(W):
        if grid[r][c] == '.':
            degree = bfs(r, c)
            max_degree = max(max_degree, degree)

print(max_degree)