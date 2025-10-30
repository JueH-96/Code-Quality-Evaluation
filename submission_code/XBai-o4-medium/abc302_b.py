H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

directions = [ (-1, -1), (-1, 0), (-1, 1),
               (0, -1),          (0, 1),
               (1, -1),  (1, 0), (1, 1) ]

for i in range(H):
    for j in range(W):
        if grid[i][j] == 's':
            for dx, dy in directions:
                x = i + 4 * dx
                y = j + 4 * dy
                if 0 <= x < H and 0 <= y < W:
                    valid = True
                    for k in range(5):
                        ni = i + k * dx
                        nj = j + k * dy
                        target = ['s', 'n', 'u', 'k', 'e'][k]
                        if grid[ni][nj] != target:
                            valid = False
                            break
                    if valid:
                        for k in range(5):
                            ni = i + k * dx
                            nj = j + k * dy
                            print(ni + 1, nj + 1)
                        exit()