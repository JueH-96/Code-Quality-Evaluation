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
            humidified_count = 0
            humidified_cells = set()
            
            for r in range(h):
                for c in range(w):
                    if grid[r][c] == '.':
                        dist1 = abs(r - floor_cells[i][0]) + abs(c - floor_cells[i][1])
                        dist2 = abs(r - floor_cells[j][0]) + abs(c - floor_cells[j][1])
                        
                        if dist1 <= d or dist2 <= d:
                            humidified_cells.add((r,c))
            
            max_humidified = max(max_humidified, len(humidified_cells))
            
    print(max_humidified)

solve()