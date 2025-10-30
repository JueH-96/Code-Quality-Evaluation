def solve():
    h, w, d = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    humidifiers = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'H':
                humidifiers.append((r, c))
    
    humidified = set()
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w and grid[r][c] != '#'
    
    def bfs(start_r, start_c):
        q = [(start_r, start_c, 0)]
        visited = set()
        
        while q:
            r, c, dist = q.pop(0)
            
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            if dist <= d:
                if grid[r][c] != '#':
                    humidified.add((r,c))
                
                
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if is_valid(nr, nc):
                        q.append((nr, nc, dist + 1))
    
    for hr, hc in humidifiers:
        bfs(hr, hc)
    
    
    count = 0
    for r, c in humidified:
        if grid[r][c] != '#':
            count += 1
    
    print(count)

solve()