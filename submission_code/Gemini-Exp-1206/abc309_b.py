n = int(input())
grid = []
for _ in range(n):
    grid.append(list(map(int, list(input()))))

outer = []
for j in range(n):
    outer.append((0, j))
for i in range(1, n):
    outer.append((i, n - 1))
for j in range(n - 2, -1, -1):
    outer.append((n - 1, j))
for i in range(n - 2, 0, -1):
    outer.append((i, 0))

new_grid = [row[:] for row in grid]
for i in range(len(outer)):
    prev_i, prev_j = outer[i - 1]
    curr_i, curr_j = outer[i]
    new_grid[curr_i][curr_j] = grid[prev_i][prev_j]

for row in new_grid:
    print("".join(map(str, row)))