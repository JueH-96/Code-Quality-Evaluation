import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:N+1]))
Q = list(map(int, data[N+1:2*N+1]))

# Create a grid filled with 0s
grid = [['0' for _ in range(N)] for _ in range(N)]

# Fill the grid according to the permutations P and Q
for i in range(N):
    for j in range(N):
        if i == P[j] - 1:
            grid[i][j] = '1'

# Print the grid
for row in grid:
    print(''.join(row))