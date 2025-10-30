def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353

    def power(a, b):
        res = 1
        a %= mod
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % mod
            a = (a * a) % mod
            b //= 2
        return res

    def modInverse(n):
        return power(n, mod - 2)

    dp = [0] * (n + 1)
    inv_n = modInverse(n)

    for i in range(n - 1, -1, -1):
        sum_a = sum(a[j] for j in range(i, n))
        sum_dp = sum(dp[j+1] for j in range(i, n))

        numerator = sum_a + sum_dp
        dp[i] = (numerator * inv_n) % mod

    print(dp[0])

solve()