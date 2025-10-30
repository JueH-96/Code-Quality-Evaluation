mod = 998244353

N, K = map(int, input().split())

inv_N = pow(N, mod - 2, mod)
term_a = (N - 2) % mod
pow_term_a = pow(term_a, K, mod)
pow_inv_N = pow(inv_N, K, mod)
a_k = (pow_term_a * pow_inv_N) % mod

term1 = ((1 - N) % mod) * a_k % mod
term2 = (N + 1) % mod
numerator = (term1 + term2) % mod

inv_2 = pow(2, mod - 2, mod)
result = (numerator * inv_2) % mod

print(result)