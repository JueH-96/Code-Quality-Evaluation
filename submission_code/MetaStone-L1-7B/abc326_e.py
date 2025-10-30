MOD = 998244353

N = int(input())
A = list(map(int, input().split()))
A = [0] + A  # Convert to 1-based indexing

inv_N = pow(N, MOD - 2, MOD)
S = [0] * (N + 2)  # S[k] for k from 0 to N

for k in range(N-1, -1, -1):
    a = A[k+1]
    s_k_plus_1 = S[k+1]
    inv_N_plus_1 = (inv_N + 1) % MOD
    term = (a + s_k_plus_1 * inv_N_plus_1) % MOD
    S[k] = term

E_0 = (S[0] * inv_N) % MOD
print(E_0)