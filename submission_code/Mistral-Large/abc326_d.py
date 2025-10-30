import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
R = data[1]
C = data[2]

def is_valid(grid, N, R, C):
    for i in range(N):
        row_count = {'A': 0, 'B': 0, 'C': 0}
        col_count = {'A': 0, 'B': 0, 'C': 0}
        for j in range(N):
            if grid[i][j] != '.':
                row_count[grid[i][j]] += 1
            if grid[j][i] != '.':
                col_count[grid[j][i]] += 1
        if row_count['A'] != 1 or row_count['B'] != 1 or row_count['C'] != 1:
            return False
        if col_count['A'] != 1 or col_count['B'] != 1 or col_count['C'] != 1:
            return False
        if grid[i][0] != R[i] and grid[i][0] != '.':
            return False
        if grid[0][i] != C[i] and grid[0][i] != '.':
            return False
    return True

def fill_grid(N, R, C):
    grid = [['.' for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if R[i] == C[j]:
                grid[i][j] = R[i]

    if is_valid(grid, N, R, C):
        return grid

    for i in range(N):
        for j in range(N):
            if grid[i][j] == '.':
                for char in 'ABC':
                    if char not in grid[i] and char not in [grid[k][j] for k in range(N)]:
                        grid[i][j] = char
                        if is_valid(grid, N, R, C):
                            return grid
                        grid[i][j] = '.'
    return None

grid = fill_grid(N, R, C)

if grid:
    print("Yes")
    for row in grid:
        print(''.join(row))
else:
    print("No")