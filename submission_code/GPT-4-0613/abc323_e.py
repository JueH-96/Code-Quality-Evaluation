MOD = 998244353
N, X = map(int, input().split())
T = list(map(int, input().split()))
T.sort()
dp = [0]*(X+1)
dp2 = [0]*(X+1)
dp[0] = 1
dp2[0] = 1
for i in range(N):
    dp, dp2 = dp2, dp
    dp2 = dp2[:]
    for j in range(X, -1, -1):
        if j < T[i]:
            dp2[j] = dp2[j]*N%MOD
        else:
            dp2[j] = (dp2[j]*N+dp[j-T[i]]*(N-1))%MOD
print(dp2[X])