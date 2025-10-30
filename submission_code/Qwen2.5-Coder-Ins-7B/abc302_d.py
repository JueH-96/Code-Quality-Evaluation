# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
D = int(data[2])

A = list(map(int, data[3:3+N]))
B = list(map(int, data[3+N:]))

A.sort()
B.sort()

max_sum = -1
i = 0
j = M - 1

while i < N and j >= 0:
    if abs(A[i] - B[j]) <= D:
        max_sum = max(max_sum, A[i] + B[j])
        i += 1
    else:
        j -= 1

print(max_sum)