from collections import deque

def solve():
    h, w, d = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    humidifiers = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'H':
                humidifiers.append((r, c))
    
    humidified = [[False] * w for _ in range(h)]
    
    q = deque()
    for r, c in humidifiers:
        q.append((r, c, 0))
        humidified[r][c] = True
    
    while q:
        r, c, dist = q.popleft()
        
        if dist == d:
            continue
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#' and not humidified[nr][nc]:
                humidified[nr][nc] = True
                q.append((nr, nc, dist + 1))
    
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.' and humidified[r][c]:
                count += 1
    
    print(count)

solve()