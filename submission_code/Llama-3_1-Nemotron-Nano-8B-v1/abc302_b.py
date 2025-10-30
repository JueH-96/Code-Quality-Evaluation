H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

directions = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

for i in range(H):
    for j in range(W):
        if grid[i][j] != 's':
            continue
        for dx, dy in directions:
            x4 = i + 4 * dx
            y4 = j + 4 * dy
            if x4 < 0 or x4 >= H or y4 < 0 or y4 >= W:
                continue
            valid = True
            for k in range(5):
                x = i + dx * k
                y = j + dy * k
                if grid[x][y] != ['s', 'n', 'u', 'k', 'e'][k]:
                    valid = False
                    break
            if valid:
                for x, y in [(i + dx * k, j + dy * k) for k in range(5)]:
                    print(x + 1, y + 1)
                exit()