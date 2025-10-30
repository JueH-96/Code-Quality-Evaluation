import sys
data = sys.stdin.read().split()
index = 0
mod = 998244353
N = int(data[index])
index += 1
Q = int(data[index])
index += 1
A = [0] * (N + 1)
for i in range(2, N + 1):
    A[i] = int(data[index])
    index += 1
inv = [0] * (N + 2)
for i in range(1, N + 2):
    inv[i] = pow(i, mod - 2, mod)
cum_A_over_c = [0] * (N + 1)
current_sum_c = 0
for x in range(2, N + 1):
    C_x = (A[x] * inv[x]) % mod
    current_sum_c = (current_sum_c + C_x) % mod
    cum_A_over_c[x] = current_sum_c
cum_A_over_shift = [0] * (N + 1)
current_sum_shift = 0
for x in range(2, N + 1):
    D_x = (A[x] * inv[x + 1]) % mod
    current_sum_shift = (current_sum_shift + D_x) % mod
    cum_A_over_shift[x] = current_sum_shift
fact_val = 1
for i in range(1, N):
    fact_val = (fact_val * i) % mod
for _ in range(Q):
    u = int(data[index])
    index += 1
    v = int(data[index])
    index += 1
    S = 0
    val1 = (-2 * cum_A_over_c[u - 1]) % mod
    S += val1
    S %= mod
    val2 = (4 * cum_A_over_shift[u - 1]) % mod
    S += val2
    S %= mod
    if u >= 2:
        A_u_mod = A[u] % mod
        inv_u_mod = inv[u]
        prod_inv_u = (A_u_mod * inv_u_mod) % mod
        A_u_minus_prod = (A_u_mod - prod_inv_u + mod) % mod
        S += A_u_minus_prod
        S %= mod
    diff_sum_c = (cum_A_over_c[v - 1] - cum_A_over_c[u] + mod) % mod
    S += diff_sum_c
    S %= mod
    A_v_mod = A[v] % mod
    S += A_v_mod
    S %= mod
    ans = (fact_val * S) % mod
    print(ans)