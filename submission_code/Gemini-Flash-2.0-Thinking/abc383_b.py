def solve():
    h, w, d = map(int, input().split())
    grid = [input() for _ in range(h)]

    floor_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                floor_cells.append((r, c))

    n_floor = len(floor_cells)
    max_humidified_cells = 0

    for i in range(n_floor):
        for j in range(i + 1, n_floor):
            r1, c1 = floor_cells[i]
            r2, c2 = floor_cells[j]

            humidified_count = 0
            for fr, fc in floor_cells:
                dist1 = abs(fr - r1) + abs(fc - c1)
                dist2 = abs(fr - r2) + abs(fc - c2)
                if dist1 <= d or dist2 <= d:
                    humidified_count += 1
            max_humidified_cells = max(max_humidified_cells, humidified_count)

    print(max_humidified_cells)

# YOUR CODE HERE
solve()