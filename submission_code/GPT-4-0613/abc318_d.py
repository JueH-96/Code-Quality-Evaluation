import itertools

N = int(input().strip())
D = [[0]*N for _ in range(N)]
for i in range(N-1):
    D[i][i+1:] = list(map(int, input().strip().split()))
    for j in range(i+1, N):
        D[j][i] = D[i][j]

dp = [0]*(1<<N)
for bit in range(1, 1<<N):
    for i, j in itertools.combinations([k for k in range(N) if (bit>>k)&1], 2):
        dp[bit] = max(dp[bit], dp[bit^(1<<i)^(1<<j)] + D[i][j])
print(dp[-1])