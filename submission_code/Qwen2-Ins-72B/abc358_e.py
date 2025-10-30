MOD = 998244353

def modpow(x, n):
    if n == 0:
        return 1
    elif n % 2 == 1:
        return x * modpow(x, n - 1) % MOD
    else:
        y = modpow(x, n // 2)
        return y * y % MOD

def modinv(x):
    return modpow(x, MOD - 2)

def comb_modp(n, k):
    if k < 0 or k > n:
        return 0
    elif k == 0 or k == n:
        return 1
    else:
        numerator = 1
        for i in range(n, n - k, -1):
            numerator = numerator * i % MOD
        denominator = modpow(k, MOD - 2)
        return numerator * denominator % MOD

def main():
    K = int(input())
    C = list(map(int, input().split()))

    dp = [[0] * 27 for _ in range(K + 1)]
    dp[0][0] = 1

    for k in range(1, K + 1):
        for i in range(26):
            for j in range(C[i] + 1):
                dp[k][j + 1] += dp[k - 1][j] * comb_modp(k, j)
                dp[k][j + 1] %= MOD

    ans = 0
    for i in range(1, 27):
        ans += dp[K][i]
        ans %= MOD

    print(ans)

if __name__ == "__main__":
    main()