H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]
directions = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]
expected = ['s', 'n', 'u', 'k', 'e']

for i in range(H):
    for j in range(W):
        if grid[i][j] != 's':
            continue
        for dx, dy in directions:
            valid = True
            for k in range(5):
                x = i + dx * k
                y = j + dy * k
                if x < 0 or x >= H or y < 0 or y >= W:
                    valid = False
                    break
            if not valid:
                continue
            # Check letters
            match = True
            for k in range(5):
                x = i + dx * k
                y = j + dy * k
                if grid[x][y] != expected[k]:
                    match = False
                    break
            if match:
                # Output the coordinates
                for step in range(5):
                    r = i + dx * step
                    c = j + dy * step
                    print(r + 1, c + 1)
                exit()