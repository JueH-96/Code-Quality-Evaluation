def solve():
    h, w, x = map(int, input().split())
    p, q = map(int, input().split())
    grid = []
    for _ in range(h):
        grid.append(list(map(int, input().split())))

    p -= 1
    q -= 1

    total_strength = grid[p][q]
    absorbed = [[False] * w for _ in range(h)]
    absorbed[p][q] = True
    
    slimes = []
    
    def get_adjacent(r, c):
        adj = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not absorbed[nr][nc]:
                adj.append((nr, nc))
        return adj

    
    current_slimes = get_adjacent(p,q)
    for r,c in current_slimes:
        slimes.append( (r,c, grid[r][c]) )

    slimes.sort(key=lambda x: x[2])

    while slimes:
        r, c, strength = slimes.pop(0)
        
        if strength * x < total_strength:
            total_strength += strength
            absorbed[r][c] = True
            new_slimes = get_adjacent(r,c)
            for nr, nc in new_slimes:
                slimes.append((nr, nc, grid[nr][nc]))
                slimes.sort(key=lambda x: x[2])
        
    print(total_strength)

solve()