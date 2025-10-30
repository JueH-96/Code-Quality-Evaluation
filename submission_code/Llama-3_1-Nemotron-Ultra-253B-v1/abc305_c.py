H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

hashes = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            hashes.append((i + 1, j + 1))

min_row = min(i for i, j in hashes)
max_row = max(i for i, j in hashes)
min_col = min(j for i, j in hashes)
max_col = max(j for i, j in hashes)

for i in range(min_row, max_row + 1):
    for j in range(min_col, max_col + 1):
        if grid[i-1][j-1] == '.':
            print(i, j)
            exit()