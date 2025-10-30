# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:]))

for i in range(1, M+1):
    C = 0
    balls = A[B[i-1]]
    A[B[i-1]] = 0
    while balls > 0:
        C += 1
        A[(B[i-1] + C) % N] += 1
        balls -= 1

print(' '.join(map(str, A)))