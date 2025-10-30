H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

for i in range(1, H + 1):
    for j in range(1, W + 1):
        if grid[i-1][j-1] == 's':
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    valid = True
                    positions = []
                    for k in range(5):
                        x = i + dx * k
                        y = j + dy * k
                        if x < 1 or x > H or y < 1 or y > W:
                            valid = False
                            break
                        positions.append((x, y))
                    if not valid:
                        continue
                    letters = [grid[x-1][y-1] for x, y in positions]
                    if letters == ['s', 'n', 'u', 'k', 'e']:
                        for x, y in positions:
                            print(x, y)
                        exit()