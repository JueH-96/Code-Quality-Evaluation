import sys

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

# Compute all possible values
values = []
for i in range(N):
    for j in range(N):
        for k in range(N):
            value = A[i] * B[j] + B[j] * C[k] + C[k] * A[i]
            values.append(value)

# Sort the values in descending order and return the Kth largest value
values.sort(reverse=True)
print(values[K-1])