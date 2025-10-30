MOD = 998244353

N, M = map(int, input().split())

pow2M = pow(2, M, MOD)
term = (pow2M - 1) * N % MOD
term = term * pow(pow2M, MOD-2, MOD) % MOD  # Modular inverse
result = (N * M + 1) * term % MOD

print(result)