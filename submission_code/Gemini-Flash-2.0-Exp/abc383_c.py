def solve():
    H, W, D = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    humidifiers = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'H':
                humidifiers.append((r, c))
    
    humidified = set()
    
    def bfs(start_r, start_c):
        q = [(start_r, start_c, 0)]
        visited = set()
        visited.add((start_r, start_c))
        
        while q:
            r, c, dist = q.pop(0)
            
            if grid[r][c] != '#':
                humidified.add((r, c))
            
            if dist < D:
                # Move up
                if r > 0 and (r-1, c) not in visited and grid[r-1][c] != '#':
                    q.append((r-1, c, dist + 1))
                    visited.add((r-1, c))
                # Move down
                if r < H - 1 and (r+1, c) not in visited and grid[r+1][c] != '#':
                    q.append((r+1, c, dist + 1))
                    visited.add((r+1, c))
                # Move left
                if c > 0 and (r, c-1) not in visited and grid[r][c-1] != '#':
                    q.append((r, c-1, dist + 1))
                    visited.add((r, c-1))
                # Move right
                if c < W - 1 and (r, c+1) not in visited and grid[r][c+1] != '#':
                    q.append((r, c+1, dist + 1))
                    visited.add((r, c+1))
    
    for r, c in humidifiers:
        bfs(r, c)
    
    count = 0
    for r, c in humidified:
        if grid[r][c] != '#':
            count += 1
            
    print(count)

solve()