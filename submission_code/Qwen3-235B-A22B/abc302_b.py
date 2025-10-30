import sys

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

letters = ['s', 'n', 'u', 'k', 'e']
directions = [
    (0, 1), (0, -1),
    (1, 0), (-1, 0),
    (1, 1), (1, -1),
    (-1, 1), (-1, -1)
]

for i in range(H):
    for j in range(W):
        if grid[i][j] != 's':
            continue
        for dx, dy in directions:
            valid = True
            for step in range(5):
                ni = i + dx * step
                nj = j + dy * step
                if ni < 0 or ni >= H or nj < 0 or nj >= W:
                    valid = False
                    break
                if grid[ni][nj] != letters[step]:
                    valid = False
                    break
            if valid:
                for step in range(5):
                    ni = i + dx * step
                    nj = j + dy * step
                    print(ni + 1, nj + 1)
                sys.exit()