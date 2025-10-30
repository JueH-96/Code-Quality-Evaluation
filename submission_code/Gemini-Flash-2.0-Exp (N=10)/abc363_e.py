def solve():
    h, w, y = map(int, input().split())
    a = []
    for _ in range(h):
        a.append(list(map(int, input().split())))

    def sink(grid, level, r, c, visited):
        if r < 0 or r >= h or c < 0 or c >= w or (r, c) in visited or grid[r][c] > level:
            return 0
        
        visited.add((r, c))
        
        count = 1
        count += sink(grid, level, r + 1, c, visited)
        count += sink(grid, level, r - 1, c, visited)
        count += sink(grid, level, r, c + 1, visited)
        count += sink(grid, level, r, c - 1, visited)
        return count

    for year in range(1, y + 1):
        sunk_count = 0
        visited = set()
        
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and a[r][c] <= year and (r,c) not in visited:
                    sunk_count += sink(a, year, r, c, visited)
        
        print(h * w - sunk_count)

solve()