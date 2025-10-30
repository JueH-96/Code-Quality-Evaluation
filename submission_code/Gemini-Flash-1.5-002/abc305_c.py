# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

min_r, max_r = H, -1
min_c, max_c = W, -1

for r in range(H):
    for c in range(W):
        if grid[r][c] == '#':
            min_r = min(min_r, r)
            max_r = max(max_r, r)
            min_c = min(min_c, c)
            max_c = max(max_c, c)

count = (max_r - min_r + 1) * (max_c - min_c + 1)
actual_count = 0
for r in range(H):
    for c in range(W):
        if grid[r][c] == '#':
            actual_count += 1

for r in range(min_r, max_r + 1):
    for c in range(min_c, max_c + 1):
        if grid[r][c] == '.':
            print(r + 1, c + 1)
            break
    else:
        continue
    break