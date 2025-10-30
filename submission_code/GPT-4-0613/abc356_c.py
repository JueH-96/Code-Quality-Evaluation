N, M, K = map(int, input().split())
tests = []
for _ in range(M):
    test = list(map(int, input().split()))
    C = test[0]
    keys = test[1:-1]
    result = test[-1]
    tests.append((C, keys, result))

dp = [[0]*16 for _ in range(1<<N)]
dp[0][0] = 1

for mask in range(1<<N):
    for b in range(N):
        if not (mask & (1<<b)):
            for k in range(16):
                if k < K:
                    dp[mask|(1<<b)][k+1] += dp[mask][k]
                else:
                    dp[mask|(1<<b)][k] += dp[mask][k]

for C, keys, result in tests:
    keys = [k-1 for k in keys]
    for mask in range(1<<N):
        cnt = sum((mask & (1<<k)) > 0 for k in keys)
        if (result == 'o' and cnt < K) or (result == 'x' and cnt >= K):
            for k in range(16):
                dp[mask][k] = 0

print(sum(dp[mask][K] for mask in range(1<<N)))