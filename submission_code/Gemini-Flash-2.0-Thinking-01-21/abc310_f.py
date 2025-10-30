def power(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

def solve():
    n = int(input())
    a_values = list(map(int, input().split()))
    mod = 998244353
    dp = [[0] * 11 for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        inv_a = power(a_values[i-1], mod - 2, mod)
        for j in range(11):
            dp[i][j] = dp[i-1][j]
            sum_term = 0
            for k in range(1, min(j, a_values[i-1]) + 1):
                sum_term = (sum_term + dp[i-1][j-k]) % mod
            dp[i][j] = (dp[i][j] + (inv_a * sum_term) % mod) % mod
            
    print(dp[n][10])

if __name__ == '__main__':
    solve()