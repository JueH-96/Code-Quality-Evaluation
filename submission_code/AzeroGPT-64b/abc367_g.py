def modexp(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res 


def initialize_xors(a):
    for i in range(19):
        for j in range(len(a)):
            if a[j] & (1 << i):
                xors[i] |= 1 << j

n, m, k = map(int, input().split())
a = list(map(int, input().split()))

MOD = 998244353
xors = [0] * 19
initialize_xors(a)
dp = [[0] * (m+1) for _ in range(19)]
for i, c_i in enumerate(xors):
    dp[i][0] = 1
    for j in range(1, m+1):
        dp[i][j] = ((dp[i][j-1] * 2 ** dp[i][0]) % MOD) * modexp(-1, (dp[i][0] + j) % m * modinv(2, MOD), MOD)

_INV_2M = modinv(2**m, MOD)

power_k = modexp(2, k, MOD)       
sum_score = 0 
for i in reversed(range(19)):
    for j in reversed(range(m)):
        if j + 1 < m:
            dp[i][j + 1] = (dp[i][j] * power_k) % MOD
        dp[i][j] = (dp[i][j] * (power_k + MOD - dp[i][(j+1) % m] * _INV_2M % MOD) * modinv(2 ** (c_i & 1), MOD)
                            * modexp(2, (j + 1) % m * modinv(2, MOD) * _INV_2M, MOD)) % MOD
    sum_score = (sum_score + dp[i][0]) % MOD
    if (i < 18):
        for j in range(m):
            dp[i+1][j] = (dp[i+1][j] + dp[i][j]) % MOD

print(sum_score)