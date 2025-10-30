# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])

grid = [['' for _ in range(N)] for _ in range(N)]

center = (N // 2, N // 2)
grid[center[0]][center[1]] = 'T'

part = 1
for i in range(N):
    for j in range(N):
        if (i, j) == center:
            continue
        grid[i][j] = str(part)
        part += 1

# Print the grid
for row in grid:
    print(' '.join(row))