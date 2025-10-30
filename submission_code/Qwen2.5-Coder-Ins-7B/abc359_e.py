# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
H = list(map(int, data[1:]))

A = [0] * (N + 1)
operations = [0] * N

for i in range(1, N + 1):
    while A[i - 1] <= A[i] or A[i - 1] <= H[i - 1]:
        A[i - 1] += 1
        A[i] -= 1
        operations[i - 1] += 1

print(' '.join(map(str, operations)))