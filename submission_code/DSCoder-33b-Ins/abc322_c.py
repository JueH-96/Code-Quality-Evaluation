import bisect

# Read inputs
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize the result list
res = [0] * N

# For each day, find the first day that has fireworks
for i in range(N):
    idx = bisect.bisect_left(A, i+1)
    if idx < M:
        res[i] = A[idx] - (i+1)

# Print the result
for r in res:
    print(r)