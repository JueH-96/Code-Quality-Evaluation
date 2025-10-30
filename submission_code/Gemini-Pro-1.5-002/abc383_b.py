def solve():
    h, w, d = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    floors = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                floors.append((r, c))
    
    max_humidified = 0
    for i in range(len(floors)):
        for j in range(i + 1, len(floors)):
            r1, c1 = floors[i]
            r2, c2 = floors[j]
            
            humidified = set()
            for r in range(h):
                for c in range(w):
                    if grid[r][c] == '.':
                        if abs(r - r1) + abs(c - c1) <= d or abs(r - r2) + abs(c - c2) <= d:
                            humidified.add((r, c))
            
            max_humidified = max(max_humidified, len(humidified))
            
    print(max_humidified)

solve()