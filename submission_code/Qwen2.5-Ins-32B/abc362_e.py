import sys
from collections import defaultdict

def solve(N, A):
    MOD = 998244353
    dp = [defaultdict(int) for _ in range(N)]
    ans = [0] * (N + 1)
    ans[1] = N

    for i in range(N):
        for j in range(i):
            diff = A[i] - A[j]
            dp[i][diff] += dp[j][diff] + 1
            ans[2] += dp[j][diff] + 1
        ans[2] %= MOD

    for k in range(3, N + 1):
        new_dp = [defaultdict(int) for _ in range(N)]
        for i in range(N):
            for j in range(i):
                diff = A[i] - A[j]
                new_dp[i][diff] += dp[j][diff]
                ans[k] += dp[j][diff]
            ans[k] %= MOD
        dp = new_dp

    print(' '.join(map(str, ans[1:])))

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    solve(N, A)