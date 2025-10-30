# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

A.sort()

max_gifts = 0
i = 0
for j in range(N):
    while i < N and A[i] < A[j] + M:
        i += 1
    max_gifts = max(max_gifts, i - j)

print(max_gifts)