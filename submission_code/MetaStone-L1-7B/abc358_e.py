MOD = 998244353

K = int(input())
C = list(map(int, input().split()))

dp = [0] * (K + 1)
dp[0] = 1  # Base case: empty string

for i in range(25, -1, -1):
    Ci = C[i]
    for m in range(K, -1, -1):
        if dp[m] == 0:
            continue
        for t in range(1, Ci + 1):
            if m + t > K:
                continue
            dp[m + t] = (dp[m + t] + dp[m]) % MOD

total = sum(dp[1:K+1]) % MOD
print(total)