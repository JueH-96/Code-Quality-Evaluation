# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

r_min = H
r_max = -1
c_min = W
c_max = -1

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            r_min = min(r_min, i)
            r_max = max(r_max, i)
            c_min = min(c_min, j)
            c_max = max(c_max, j)

for i in range(r_min, r_max+1):
    for j in range(c_min, c_max+1):
        if grid[i][j] == '.':
            print(i+1, j+1)
            exit()