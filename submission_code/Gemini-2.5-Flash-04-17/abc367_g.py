import sys

def read_int():
    return int(sys.stdin.readline())

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def read_longs():
    return list(map(int, sys.stdin.readline().split()))

MOD = 998244353
V = 1 << 20 # Max possible XOR sum is < 2^20, so use V=2^20

def fwht(a, inverse):
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                u = a[i + j]
                v = a[i + h + j]
                a[i + j] = (u + v) % MOD
                a[i + h + j] = (u - v) % MOD
        h *= 2

    if inverse:
        inv_n = pow(n, MOD - 2, MOD)
        for i in range(n):
            a[i] = (a[i] * inv_n) % MOD

# Polynomial multiplication modulo y^M - 1 (Cyclic Convolution)
# Input polynomials p1, p2 represented by list of coefficients [c0, c1, ..., c_m-1]
# Output polynomial modulo y^M - 1
def poly_mul_mod_ym(p1, p2, M):
    # Ensure polynomials have M coefficients for cyclic convolution perspective
    a = list(p1)
    b = list(p2)
    while len(a) < M:
        a.append(0)
    while len(b) < M:
        b.append(0)
    
    res_coeffs = [0] * M
    for i in range(M):
        if a[i] == 0: continue
        for j in range(M):
            if b[j] == 0: continue
            res_coeffs[(i + j) % M] = (res_coeffs[(i + j) % M] + a[i] * b[j]) % MOD
    return res_coeffs

# Polynomial exponentiation modulo y^M - 1
# Base is a polynomial represented by list of coefficients
def poly_pow_mod_ym(base, exp, M):
    res = [0] * M
    res[0] = 1 # Identity element is 1
    
    base_m = list(base) # Ensure base has M coefficients
    while len(base_m) < M:
        base_m.append(0)

    while exp > 0:
        if exp % 2 == 1:
            res = poly_mul_mod_ym(res, base_m, M)
        base_m = poly_mul_mod_ym(base_m, base_m, M)
        exp //= 2
    return res

def main():
    N, M, K = read_ints()
    A = read_ints()

    # 1. Compute B_x: counts of each value in A
    B = [0] * V
    for x in A:
        B[x] += 1

    # 2. Compute FWHT of B
    hat_B = list(B)
    fwht(hat_B, False)

    # 3. Compute S_K(k) = FWHT of f(x) = x^K
    f = [0] * V
    for x in range(V):
        f[x] = pow(x, K, MOD)
    S_K = list(f)
    fwht(S_K, False)

    # 4. Compute g(d) = [y^0]((1+y)^{(N+d)/2}(1-y)^{(N-d)/2} mod y^M-1)
    # We only need g(d) for values of d that appear in hat_B.
    # The possible values of d = hat_B_k are integers.
    # hat_B_k = N_0(k) - N_1(k), where N_0(k) is count of A_i with k.A_i even, N_1(k) count with k.A_i odd.
    # N_0(k) + N_1(k) = N. So hat_B_k must have same parity as N, and |hat_B_k| <= N.
    # We only need to compute g(d) for d in [-N, N] with same parity as N.

    g_values = {} # Map d to g(d)

    one_plus_y = [0] * M
    one_plus_y[0] = 1
    one_plus_y[1] = 1

    one_minus_y = [0] * M
    one_minus_y[0] = 1
    one_minus_y[1] = (MOD - 1) % MOD

    # Iterate over possible values of d = N_0 - N_1
    # N_0 + N_1 = N, N_0, N_1 >= 0 integers.
    # d = N_0 - (N - N_0) = 2*N_0 - N.
    # N_0 can range from 0 to N.
    # This covers all integers d from -N to N with the same parity as N.
    
    for N0 in range(N + 1):
        N1 = N - N0
        d = N0 - N1
        
        # Compute (1+y)^N0 mod y^M-1
        poly_N0_term = poly_pow_mod_ym(one_plus_y, N0, M)
        
        # Compute (1-y)^N1 mod y^M-1
        poly_N1_term = poly_pow_mod_ym(one_minus_y, N1, M)
        
        # Compute their product mod y^M-1
        prod_poly = poly_mul_mod_ym(poly_N0_term, poly_N1_term, M)
        
        # g(d) is the constant term [y^0]
        g_d = prod_poly[0]
        g_values[d] = g_d
            
    # Total sum is V_inv * sum_k S_K(k) * (R_0(k) - 1)
    # where R_0(k) = g(hat_B_k).
    
    total_weighted_sum_R0 = 0
    total_sum_SK = 0 # Sum of S_K(k) for all k. Used to subtract 1 for each k.

    for k in range(V):
        d = hat_B[k]
        R_0_k = g_values[d]
        
        # We need sum over non-empty subsequences.
        # R_0(k) represents sum over l = 0, M, 2M, ...
        # hat_C(0, k) = [y^0] hat_F(y,k) = 1.
        # Summand should be Sum_{l=1, l=0(M)} hat_C(l, k). This is R_0(k) - hat_C(0, k) = R_0(k) - 1.
        
        # Sum_k S_K(k) (R_0(k) - 1) = Sum_k S_K(k) R_0(k) - Sum_k S_K(k)
        
        total_weighted_sum_R0 = (total_weighted_sum_R0 + S_K[k] * R_0_k) % MOD
        total_sum_SK = (total_sum_SK + S_K[k]) % MOD

    inv_V = pow(V, MOD - 2, MOD)
    
    # Final sum is V_inv * (total_weighted_sum_R0 - total_sum_SK)
    # total_sum_SK = sum_k hat_f_k. By FWHT property, sum_k hat_f_k = V * f_0 = V * 0^K.
    # Since K >= 1, 0^K = 0. So total_sum_SK is 0.
    # This term can be ignored because it cancels out if K >= 1. Problem constraints say K >= 1.
    
    ans = (total_weighted_sum_R0 * inv_V) % MOD

    print(ans)

if __name__ == "__main__":
    main()