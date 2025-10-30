def find_snake(grid, H, W):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for dx, dy in directions:
                    try:
                        if (grid[i + dx][j + dy] == 'n' and
                            grid[i + 2*dx][j + 2*dy] == 'u' and
                            grid[i + 3*dx][j + 3*dy] == 'k' and
                            grid[i + 4*dx][j + 4*dy] == 'e' and
                            i + 4*dx < H and j + 4*dy < W and
                            j + 4*dy >= 0):
                            return [(i + 1, j + 1),
                                    (i + dx + 1, j + dy + 1),
                                    (i + 2*dx + 1, j + 2*dy + 1),
                                    (i + 3*dx + 1, j + 3*dy + 1),
                                    (i + 4*dx + 1, j + 4*dy + 1)]
                    except IndexError:
                        pass
    return []

H, W = map(int, input().split())
grid = [input() for _ in range(H)]

snake_cells = find_snake(grid, H, W)

for cell in snake_cells:
    print(*cell)