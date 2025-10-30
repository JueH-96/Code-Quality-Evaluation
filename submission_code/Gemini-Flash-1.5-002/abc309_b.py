# YOUR CODE HERE
n = int(input())
grid = []
for _ in range(n):
    grid.append(list(map(int, list(input()))))

temp = []
for i in range(n):
    for j in range(n):
        if i == 0 or i == n - 1 or j == 0 or j == n - 1:
            temp.append((i, j))

shifted = [0] * len(temp)
for i in range(len(temp)):
    x, y = temp[i]
    if x == 0 and y < n - 1:
        shifted[i] = grid[x][y + 1]
    elif x < n - 1 and y == n - 1:
        shifted[i] = grid[x + 1][y]
    elif x == n - 1 and y > 0:
        shifted[i] = grid[x][y - 1]
    elif y == 0 and x > 0:
        shifted[i] = grid[x - 1][y]
    else:
        shifted[i] = grid[x][y]

k = 0
for i in range(n):
    for j in range(n):
        if i == 0 or i == n - 1 or j == 0 or j == n - 1:
            grid[i][j] = shifted[k]
            k += 1

for row in grid:
    print(''.join(map(str, row)))