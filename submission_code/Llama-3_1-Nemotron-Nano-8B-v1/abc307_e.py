MOD = 998244353
N, M = map(int, input().split())

term1 = pow(M - 1, N, MOD)
coeff = 1 if N % 2 == 0 else -1
term2 = (coeff * (M - 1)) % MOD
ans = (term1 + term2) % MOD

print(ans)