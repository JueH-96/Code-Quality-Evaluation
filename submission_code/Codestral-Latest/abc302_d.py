# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
D = int(data[index])
index += 1

A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + M]))

A.sort()
B.sort()

max_sum = -1
j = M - 1

for i in range(N):
    while j >= 0 and A[i] - B[j] > D:
        j -= 1
    if j >= 0 and abs(A[i] - B[j]) <= D:
        max_sum = max(max_sum, A[i] + B[j])

print(max_sum)