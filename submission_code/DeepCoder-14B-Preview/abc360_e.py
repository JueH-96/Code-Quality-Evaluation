MOD = 998244353

N, K = map(int, input().split())

if N == 1:
    print(1)
else:
    inv_2 = pow(2, MOD - 2, MOD)
    inv_N = pow(N, MOD - 2, MOD)
    a_base = ((N - 2) * inv_N) % MOD
    a_pow_k = pow(a_base, K, MOD)
    term1 = ((1 - N) % MOD) * a_pow_k % MOD
    term1 = term1 * inv_2 % MOD
    term2 = ((N + 1) % MOD) * inv_2 % MOD
    E_K = (term1 + term2) % MOD
    print(E_K)