def solve():
    H, W, X = map(int, input().split())
    P, Q = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(map(int, input().split())))

    takahashi_strength = grid[P-1][Q-1]
    
    visited = set()
    visited.add((P-1, Q-1))
    
    q = [(P-1, Q-1)]
    
    while q:
        r, c = q.pop(0)
        
        neighbors = []
        if r > 0:
            neighbors.append((r-1, c))
        if r < H - 1:
            neighbors.append((r+1, c))
        if c > 0:
            neighbors.append((r, c-1))
        if c < W - 1:
            neighbors.append((r, c+1))
            
        for nr, nc in neighbors:
            if grid[nr][nc] < (takahashi_strength / X):
                if (nr, nc) not in visited:
                    takahashi_strength += grid[nr][nc]
                    visited.add((nr, nc))
                    q.append((nr, nc))
                    
    print(takahashi_strength)

solve()