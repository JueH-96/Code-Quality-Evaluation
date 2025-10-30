import sys
data = sys.stdin.read().split()
H = int(data[0])
W = int(data[1])
grid = data[2:2+H]
min_row = H
max_row = -1
min_col = W
max_col = -1
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            if i < min_row:
                min_row = i
            if i > max_row:
                max_row = i
            if j < min_col:
                min_col = j
            if j > max_col:
                max_col = j
for i in range(min_row, max_row + 1):
    for j in range(min_col, max_col + 1):
        if grid[i][j] == '.':
            print(i + 1, j + 1)