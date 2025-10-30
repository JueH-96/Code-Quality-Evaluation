n = int(input())
grid = [list(input().strip()) for _ in range(n)]

for i in range(1, (n // 2) + 1):
    changes = []
    for x_1 in range(i, (n + 1 - i) + 1):
        for y_1 in range(i, (n + 1 - i) + 1):
            # Calculate target row and column in 0-based indices
            target_row = y_1 - 1
            target_col = n - x_1
            # Get the value from the original position
            value = grid[x_1 - 1][y_1 - 1]
            changes.append((target_row, target_col, value))
    # Apply all changes
    for tr, tc, v in changes:
        grid[tr][tc] = v

for row in grid:
    print(''.join(row))