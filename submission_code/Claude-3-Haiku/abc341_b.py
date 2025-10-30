import sys

# Read input
N = int(input())
A = list(map(int, input().split()))
S_T = []
for _ in range(N-1):
    s, t = map(int, input().split())
    S_T.append((s, t))

# Perform the operations
for i in range(N-1):
    s, t = S_T[i]
    if A[i] >= s:
        A[i] -= s
        A[i+1] += t

# Print the maximum possible number of units of the currency of country N
print(A[-1])