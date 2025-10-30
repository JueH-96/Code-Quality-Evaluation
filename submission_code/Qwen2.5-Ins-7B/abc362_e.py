# YOUR CODE HERE
from collections import defaultdict

def solve():
    mod = 998244353
    n = int(input())
    a = list(map(int, input().split()))
    dp = [defaultdict(int) for _ in range(n)]
    ans = [0] * n
    
    for i in range(n):
        dp[i][a[i]] = 1
        ans[0] += 1
    
    for k in range(1, n):
        for i in range(n):
            for j in range(i):
                diff = a[i] - a[j]
                dp[i][diff] += dp[j][diff]
                dp[i][diff] %= mod
            ans[k] += dp[i][a[i]]
            ans[k] %= mod
    
    print(*ans)

solve()