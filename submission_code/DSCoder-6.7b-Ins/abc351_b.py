# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
grid_A = [sys.stdin.readline().strip() for _ in range(N)]
grid_B = [sys.stdin.readline().strip() for _ in range(N)]

for i in range(N):
    for j in range(N):
        if grid_A[i][j] != grid_B[i][j]:
            print(i+1, j+1)
            break