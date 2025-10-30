# YOUR CODE HERE
from collections import defaultdict

mod = 998244353

def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    dp = [defaultdict(int) for _ in range(m)]
    for x in a:
        new_dp = [defaultdict(int) for _ in range(m)]
        for i in range(m):
            for y, cnt in dp[i].items():
                new_dp[i][y ^ x] += cnt
                new_dp[i][y] += cnt
                if (i + 1) % m == 0:
                    new_dp[i][y] += 1
        dp = new_dp
    ans = 0
    for i in range(m):
        for y, cnt in dp[i].items():
            if (i + 1) % m == 0:
                ans += pow(y, k, mod) * cnt
                ans %= mod
    print(ans)

solve()