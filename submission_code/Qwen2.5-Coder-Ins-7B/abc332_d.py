# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])

A = []
B = []

index = 2
for i in range(H):
    row = list(map(int, data[index:index+W]))
    A.append(row)
    index += W

index += W
for i in range(H):
    row = list(map(int, data[index:index+W]))
    B.append(row)
    index += W

def get_row_index(grid, row):
    for i in range(H):
        if grid[i] == row:
            return i
    return -1

def get_col_index(grid, col):
    for i in range(W):
        if [grid[j][i] for j in range(H)] == col:
            return i
    return -1

def swap_rows(grid, i, j):
    grid[i], grid[j] = grid[j], grid[i]

def swap_cols(grid, i, j):
    for k in range(H):
        grid[k][i], grid[k][j] = grid[k][j], grid[k][i]

def min_operations(A, B):
    operations = 0
    for i in range(W):
        col_A = [A[j][i] for j in range(H)]
        col_B = [B[j][i] for j in range(H)]
        if col_A != col_B:
            col_B_index = get_col_index(A, col_B)
            if col_B_index == -1:
                return -1
            swap_cols(A, i, col_B_index)
            operations += abs(i - col_B_index)
    return operations

result = min_operations(A, B)
print(result)