def solve():
    H, W, D = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    floor_cells = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '.':
                floor_cells.append((r, c))
    
    max_humidified = 0
    for i in range(len(floor_cells)):
        for j in range(i + 1, len(floor_cells)):
            r1, c1 = floor_cells[i]
            r2, c2 = floor_cells[j]
            
            humidified_count = 0
            for r in range(H):
                for c in range(W):
                    if grid[r][c] == '.':
                        dist1 = abs(r - r1) + abs(c - c1)
                        dist2 = abs(r - r2) + abs(c - c2)
                        if dist1 <= D or dist2 <= D:
                            humidified_count += 1
            
            max_humidified = max(max_humidified, humidified_count)
    
    print(max_humidified)

solve()