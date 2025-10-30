H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Collect all positions with a cookie
cookies = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            cookies.append((i + 1, j + 1))  # Convert to 1-based indices

# Determine the boundaries of the original rectangle
min_row = min(c[0] for c in cookies)
max_row = max(c[0] for c in cookies)
min_col = min(c[1] for c in cookies)
max_col = max(c[1] for c in cookies)

# Find the missing cookie within the original rectangle
for i in range(min_row, max_row + 1):
    for j in range(min_col, max_col + 1):
        # Check if the current cell is empty
        if grid[i - 1][j - 1] == '.':
            print(i, j)
            exit()