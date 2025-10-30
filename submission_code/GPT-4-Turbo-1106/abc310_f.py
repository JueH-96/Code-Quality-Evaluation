MOD = 998244353

def power(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def mod_inverse(x, p):
    return power(x, p - 2, p)

def solve(N, A):
    dp = [0] * 11
    dp[0] = 1
    for a in A:
        next_dp = [0] * 11
        for i in range(11):
            next_dp[i] = dp[i] * (a + 1)
            if i - a >= 0:
                next_dp[i] += dp[i - a] * (a - max(0, a - 10))
            next_dp[i] %= MOD
        dp = next_dp
    total_ways = power(10, N, MOD)
    return (dp[10] * mod_inverse(total_ways, MOD)) % MOD

def main():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))
    print(solve(N, A))

if __name__ == "__main__":
    main()