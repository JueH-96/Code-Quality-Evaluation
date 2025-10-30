# YOUR CODE HERE
n = int(input())
a_grid = [input() for _ in range(n)]
b_grid = [input() for _ in range(n)]

for i in range(n):
    for j in range(n):
        if a_grid[i][j] != b_grid[i][j]:
            print(i + 1, j + 1)
            break