import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
P = list(map(int, data[index:index + N]))
index += N
Q = list(map(int, data[index:index + N]))

grid = [['0'] * N for _ in range(N)]

# Fill the grid based on the given permutations P and Q
for i in range(N):
    for j in range(N):
        if P[i] < Q[j]:
            grid[i][j] = '0'
        else:
            grid[i][j] = '1'

# Print the grid
for row in grid:
    print(''.join(row))