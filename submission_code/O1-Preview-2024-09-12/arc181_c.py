# YOUR CODE HERE
import sys
N_and_rest = sys.stdin.read().split()

N = int(N_and_rest[0])
P = list(map(int, N_and_rest[1:N+1]))
Q = list(map(int, N_and_rest[N+1:2*N+1]))

# Adjust indices to zero-based
P = [x - 1 for x in P]
Q = [x - 1 for x in Q]

N = int(N)

A = [['0']*N for _ in range(N)]

for i in range(N):
    for j in range(N):
        if P[i] + Q[j] + 2 > N:
            A[i][j] = '1'

for row in A:
    print(''.join(row))