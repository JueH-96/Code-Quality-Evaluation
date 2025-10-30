# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = [list(data[i+2]) for i in range(H)]

def mark_and_remove(grid):
    marked = False
    # Mark rows
    for i in range(H):
        for j in range(W-1):
            if grid[i][j] == grid[i][j+1] and grid[i][j] != '.':
                for k in range(W):
                    if grid[i][k] != '.' and grid[i][k] != grid[i][j]:
                        break
                else:
                    for k in range(W):
                        if grid[i][k] != '.':
                            grid[i][k] = '.'
                            marked = True
    # Mark columns
    for j in range(W):
        for i in range(H-1):
            if grid[i][j] == grid[i+1][j] and grid[i][j] != '.':
                for k in range(H):
                    if grid[k][j] != '.' and grid[k][j] != grid[i][j]:
                        break
                else:
                    for k in range(H):
                        if grid[k][j] != '.':
                            grid[k][j] = '.'
                            marked = True
    return marked

while mark_and_remove(grid):
    pass

remaining_cookies = sum(row.count('.') for row in grid)
print(remaining_cookies)