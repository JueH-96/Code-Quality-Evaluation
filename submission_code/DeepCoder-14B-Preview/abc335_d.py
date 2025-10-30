n = int(input())
mid = (n - 1) // 2
grid = [[0 for _ in range(n)] for _ in range(n)]
grid[mid][mid] = 'T'
current = 1

for l in range(mid):
    top = l
    bottom = n - 1 - l
    left = l
    right = n - 1 - l

    # Fill top row
    for j in range(left, right + 1):
        if grid[top][j] == 0:
            grid[top][j] = current
            current += 1

    # Fill right column
    for i in range(top + 1, bottom + 1):
        if grid[i][right] == 0:
            grid[i][right] = current
            current += 1

    # Fill bottom row if it's a different row than top
    if top < bottom:
        for j in range(right - 1, left - 1, -1):
            if grid[bottom][j] == 0:
                grid[bottom][j] = current
                current += 1

    # Fill left column if it's a different column than right
    if left < right:
        for i in range(bottom - 1, top, -1):
            if grid[i][left] == 0:
                grid[i][left] = current
                current += 1

for row in grid:
    print(' '.join(map(str, row)))