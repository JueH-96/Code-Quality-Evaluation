n, m = map(int, input().split())
MOD = 998244353
term1 = pow(m-1, n, MOD)
sign = 1 if n % 2 == 0 else -1
term2 = (sign * (m-1)) % MOD
print((term1 + term2) % MOD)