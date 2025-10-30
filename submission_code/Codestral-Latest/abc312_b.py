# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
grid = data[2:N+2]

def is_tak_code(i, j):
    # Check the top-left 3x3 region
    if not (grid[i][j:j+3] == '###' and grid[i+1][j:j+3] == '###' and grid[i+2][j:j+3] == '###'):
        return False

    # Check the bottom-right 3x3 region
    if not (grid[i+6][j+6:j+9] == '###' and grid[i+7][j+6:j+9] == '###' and grid[i+8][j+6:j+9] == '###'):
        return False

    # Check the adjacent cells
    if not (grid[i][j+3] == '.' and grid[i+1][j+3] == '.' and grid[i+2][j+3] == '.' and
            grid[i+3][j] == '.' and grid[i+3][j+1] == '.' and grid[i+3][j+2] == '.' and grid[i+3][j+3] == '.' and
            grid[i+3][j+4] == '.' and grid[i+3][j+5] == '.' and grid[i+3][j+6] == '.' and
            grid[i+4][j] == '.' and grid[i+4][j+1] == '.' and grid[i+4][j+2] == '.' and grid[i+4][j+3] == '.' and
            grid[i+4][j+4] == '.' and grid[i+4][j+5] == '.' and grid[i+4][j+6] == '.' and
            grid[i+5][j] == '.' and grid[i+5][j+1] == '.' and grid[i+5][j+2] == '.' and grid[i+5][j+3] == '.' and
            grid[i+5][j+4] == '.' and grid[i+5][j+5] == '.' and grid[i+5][j+6] == '.' and
            grid[i+6][j+3] == '.' and grid[i+7][j+3] == '.' and grid[i+8][j+3] == '.'):
        return False

    if not (grid[i][j+5] == '.' and grid[i+1][j+5] == '.' and grid[i+2][j+5] == '.' and
            grid[i+3][j+6] == '.' and grid[i+4][j+6] == '.' and grid[i+5][j+6] == '.'):
        return False

    return True

results = []
for i in range(N-8):
    for j in range(M-8):
        if is_tak_code(i, j):
            results.append((i+1, j+1))

for result in sorted(results):
    print(result[0], result[1])