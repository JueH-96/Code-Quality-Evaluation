# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = data[1:N+1]
B = data[N+1:]

for i in range(N):
    for j in range(N):
        if A[i][j] != B[i][j]:
            print(i + 1, j + 1)
            break