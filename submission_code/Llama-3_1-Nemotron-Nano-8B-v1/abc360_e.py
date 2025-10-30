MOD = 998244353

N, K = map(int, input().split())

inv_N = pow(N, MOD - 2, MOD)
r = (N - 2) * inv_N % MOD
r_pow = pow(r, K, MOD)

term = ( (1 - N) * inv_N ) % MOD
term = term * r_pow % MOD
term = term * pow(2, MOD-2, MOD) % MOD  # divided by 2

half_N_plus_1 = (N + 1) * pow(2, MOD-2, MOD) % MOD
result = (half_N_plus_1 + term) % MOD

print(result)