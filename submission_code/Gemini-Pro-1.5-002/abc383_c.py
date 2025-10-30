import collections

def solve():
    h, w, d = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    humidifiers = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'H':
                humidifiers.append((r, c))
    
    humidified = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'H':
                humidified.add((r,c))

    for hr, hc in humidifiers:
        q = collections.deque([(hr, hc, 0)])
        visited = set([(hr, hc)])
        
        while q:
            r, c, dist = q.popleft()
            
            if dist == d:
                continue

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc, dist + 1))
                    if grid[nr][nc] == '.':
                        humidified.add((nr, nc))
    
    count = 0
    for r in range(h):
        for c in range(w):
            if (r,c) in humidified and grid[r][c] == '.':
                count += 1
                
    print(count)

solve()