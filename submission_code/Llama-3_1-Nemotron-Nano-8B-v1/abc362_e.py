import sys
from collections import defaultdict

n = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))
MOD = 998244353

dp = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
ans = [0] * n
ans[0] = n  # k=1

for i in range(n):
    temp = defaultdict(lambda: defaultdict(int))
    for j in range(i):
        d = A[i] - A[j]
        temp[d][2] = (temp[d][2] + 1) % MOD
        if d in dp[j]:
            for m in dp[j][d]:
                new_m = m + 1
                temp[d][new_m] = (temp[d][new_m] + dp[j][d][m]) % MOD
    for d in temp:
        for m in temp[d]:
            dp[i][d][m] = (dp[i][d][m] + temp[d][m]) % MOD

for k in range(2, n+1):
    total = 0
    for i in range(n):
        for d in dp[i]:
            total = (total + dp[i][d].get(k, 0)) % MOD
    ans[k-1] = total

print(' '.join(map(str, ans)))