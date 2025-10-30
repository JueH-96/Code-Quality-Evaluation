# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
grid_A = input[1:N+1]
grid_B = input[N+1:]

for i in range(N):
    for j in range(N):
        if grid_A[i][j] != grid_B[i][j]:
            print(i + 1, j + 1)
            exit()