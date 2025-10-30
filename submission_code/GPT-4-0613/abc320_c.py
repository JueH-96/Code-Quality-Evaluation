M = int(input().strip())
S = [input().strip() for _ in range(3)]

INF = 10**18
dp = [[INF]*M for _ in range(3)]
for i in range(3):
    for j in range(M):
        dp[i][j] = min(dp[i][j], j)
        for k in range(j):
            if S[i][j] == S[i][k]:
                dp[i][j] = min(dp[i][j], dp[i][k]+M)

ans = INF
for i in range(10):
    i = str(i)
    if i in S[0] and i in S[1] and i in S[2]:
        tmp = 0
        for j in range(3):
            tmp += min(dp[j][k] for k in range(M) if S[j][k] == i)
        ans = min(ans, tmp)
if ans == INF:
    print(-1)
else:
    print(ans)