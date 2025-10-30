# YOUR CODE HERE

N, K = map(int, input().split())
A = list(map(int, input().split()))

A.sort()

min_val = float('inf')

for i in range(K+1):
    if i + N - K - 1 < N:
        min_val = min(min_val, A[i + N - K - 1] - A[i])

print(min_val)