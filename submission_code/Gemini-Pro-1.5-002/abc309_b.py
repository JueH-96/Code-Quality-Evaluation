# YOUR CODE HERE
n = int(input())
grid = []
for _ in range(n):
    grid.append(list(map(int, list(input()))))

new_grid = [[0] * n for _ in range(n)]

for j in range(n):
    new_grid[0][j] = grid[0][(j - 1 + n) % n]

for i in range(1, n):
    new_grid[i][n - 1] = grid[i - 1][n - 1]

for j in range(n - 2, -1, -1):
    new_grid[n - 1][j] = grid[n - 1][(j + 1) % n]

for i in range(n - 2, 0, -1):
    new_grid[i][0] = grid[i + 1][0]


for row in new_grid:
    print("".join(map(str, row)))