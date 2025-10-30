n = int(input())
grid = [list(input().strip()) for _ in range(n)]

positions = []
# Top row left to right
for j in range(n):
    positions.append((0, j))
# Right column top to bottom (excluding top)
for i in range(1, n):
    positions.append((i, n-1))
# Bottom row right to left (excluding right)
for j in range(n-2, -1, -1):
    positions.append((n-1, j))
# Left column bottom to top (excluding bottom and top)
for i in range(n-2, 0, -1):
    positions.append((i, 0))

original = [grid[i][j] for (i, j) in positions]
rotated = [original[-1]] + original[:-1]

for k in range(len(positions)):
    i, j = positions[k]
    grid[i][j] = rotated[k]

for row in grid:
    print(''.join(row))