N, K = map(int, input().split())
A = list(map(int, input().split()))

bottom_k = A[N-K:]
rest = A[:N-K]

result = bottom_k + rest

print(*result)