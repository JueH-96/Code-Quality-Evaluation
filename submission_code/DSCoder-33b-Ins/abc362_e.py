# YOUR CODE HERE
import sys

MOD = 998244353
N = int(input())
A = list(map(int, input().split()))

dp = [[0] * 100 for _ in range(N)]
cnt = [0] * 100
ans = [0] * N

for i in range(N):
    for j in range(i):
        diff = (A[i] - A[j] + MOD) % MOD
        dp[i][diff] = (dp[i][diff] + cnt[j] + 1) % MOD
        ans[i] = (ans[i] + dp[j][diff]) % MOD
    cnt[i] = (cnt[i] + 1) % MOD
    ans[i] = (ans[i] + ans[i - 1] + 1) % MOD

print(*ans)