N, K = map(int, input().split())
A = list(map(int, input().split()))

A.sort()

ans = float('inf')
for i in range(N-K):
    ans = min(ans, A[i+K-1] - A[i])

print(ans)