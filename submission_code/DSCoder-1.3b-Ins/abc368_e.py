N, M, X = map(int, input().split())
A, B, S, T = [], [], [], []
for _ in range(M):
    a, b, s, t = map(int, input().split())
    A.append(a)
    B.append(b)
    S.append(s)
    T.append(t)

dp = [float('inf')] * (X + 1)
dp[0] = 0
for i in range(N):
    for j in range(X, A[i] - 1, -1):
        dp[j] = min(dp[j], dp[j - A[i]] + 1)

ans = [0] * M
for i in range(M):
    if dp[T[i]] <= S[i]:
        ans[i] = dp[T[i]] - S[i]

print(*ans)