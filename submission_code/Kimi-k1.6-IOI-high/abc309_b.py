n = int(input())
grid = []
for _ in range(n):
    line = input().strip()
    grid.append(list(line))

positions = []
# Top row left to right
for col in range(n):
    positions.append((0, col))
# Right column top+1 to bottom-1
for row in range(1, n-1):
    positions.append((row, n-1))
# Bottom row right to left
for col in range(n-1, -1, -1):
    positions.append((n-1, col))
# Left column bottom-1 to top+1
for row in range(n-2, 0, -1):
    positions.append((row, 0))

values = [grid[i][j] for i, j in positions]
if values:
    new_values = [values[-1]] + values[:-1]
else:
    new_values = values

for idx, (i, j) in enumerate(positions):
    grid[i][j] = new_values[idx]

for row in grid:
    print(''.join(row))