def solve():
    h, w, d = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    floor_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                floor_cells.append((r, c))
    
    max_humidified = 0
    for i in range(len(floor_cells)):
        for j in range(i + 1, len(floor_cells)):
            humidified_cells = set()
            
            r1, c1 = floor_cells[i]
            r2, c2 = floor_cells[j]
            
            humidified_cells.add((r1, c1))
            humidified_cells.add((r2, c2))
            
            for r in range(h):
                for c in range(w):
                    if grid[r][c] == '.':
                        if abs(r - r1) + abs(c - c1) <= d or abs(r - r2) + abs(c - c2) <= d:
                            humidified_cells.add((r,c))
            max_humidified = max(max_humidified, len(humidified_cells))
    print(max_humidified)

solve()