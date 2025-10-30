MOD = 998244353

N, M = map(int, input().split())

a = pow(M-1, N, MOD)
sign = pow(-1, N, MOD)
term2 = (sign * (M-1)) % MOD
ans = (a + term2) % MOD
print(ans)