import sys

# Modulo constant
MOD = 998244353
# Primitive root for NTT
PRIMITIVE_ROOT = 3

def power(base, exp):
    res = 1
    base %= MOD
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % MOD
        base = (base * base) % MOD
        exp //= 2
    return res

def inverse(n):
    return power(n, MOD - 2)

# Precompute factorials and inverse factorials
# MAX_N is chosen based on problem constraints + some buffer
MAX_N = 200005
fact = [1] * (MAX_N + 1)
inv_fact = [1] * (MAX_N + 1)
for i in range(2, MAX_N + 1):
    fact[i] = (fact[i - 1] * i) % MOD
inv_fact[MAX_N] = inverse(fact[MAX_N])
for i in range(MAX_N - 1, 1, -1):
    inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

def nCr_mod_p(n, r):
    if r < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    if r > n // 2:
        r = n - r
    return (((fact[n] * inv_fact[r]) % MOD) * inv_fact[n - r]) % MOD

# NTT implementation
def ntt(a, invert):
    n = len(a)
    j = 0
    for i in range(1, n):
        if i < j:
            a[i], a[j] = a[j], a[i]
        k = n // 2
        while (j & k) != 0:
            j ^= k
            k //= 2
        j ^= k

    for len_half in range(1, n // 2 + 1):
        len_full = len_half * 2
        wlen = power(PRIMITIVE_ROOT, (MOD - 1) // len_full)
        if invert:
            wlen = inverse(wlen)

        for i in range(0, n, len_full):
            w = 1
            for j in range(len_half):
                u = a[i + j]
                v = (a[i + j + len_half] * w) % MOD
                a[i + j] = (u + v) % MOD
                a[i + j + len_half] = (u - v + MOD) % MOD
                w = (w * wlen) % MOD

    if invert:
        inv_n = inverse(n)
        for i in range(n):
            a[i] = (a[i] * inv_n) % MOD

def poly_mul(a, b):
    deg_a = len(a) - 1
    deg_b = len(b) - 1
    size = 1
    while size <= deg_a + deg_b:
        size *= 2
    
    a_padded = list(a) + [0] * (size - len(a))
    b_padded = list(b) + [0] * (size - len(b))

    ntt(a_padded, False)
    ntt(b_padded, False)

    c_padded = [(a_padded[i] * b_padded[i]) % MOD for i in range(size)]

    ntt(c_padded, True)
    
    result_deg = deg_a + deg_b
    return c_padded[:result_deg + 1]

# Function to compute coefficients of (1+ax)^m up to max_deg
def poly_binom_power(a, m, max_deg):
    coeffs = [0] * (max_deg + 1)
    a_mod = a % MOD
    pow_a_j = 1
    for j in range(min(m, max_deg) + 1):
        term = (nCr_mod_p(m, j) * pow_a_j) % MOD
        coeffs[j] = term
        pow_a_j = (pow_a_j * a_mod) % MOD
    return coeffs

# Recursive function to compute P(x) coefficients
# poly_terms: list of (a, m) where the polynomial is (1+ax)^m
def compute_P_coeffs(poly_terms):
    if len(poly_terms) == 0:
        return [1] # Identity for multiplication is polynomial 1
    if len(poly_terms) == 1:
        a_val, m_val = poly_terms[0]
        # Max degree needed is N for P(x)
        return poly_binom_power(a_val, m_val, N)

    mid = len(poly_terms) // 2
    poly_left = compute_P_coeffs(poly_terms[:mid])
    poly_right = compute_P_coeffs(poly_terms[mid:])

    return poly_mul(poly_left, poly_right)

def get_num_digits(n):
    if n == 0: return 1
    count = 0
    temp = n
    while temp > 0:
        temp //= 10
        count += 1
    return count

N = int(sys.stdin.readline())

# Calculate counts of numbers with d digits in {1, ..., N}
D = get_num_digits(N)
C_d = {}
powers_of_10_val = [1]
for _ in range(D + 1): # Need powers up to 10^D
    powers_of_10_val.append(powers_of_10_val[-1] * 10)

for d in range(1, D + 1):
    # Count of numbers with d digits in {1, ..., N}
    # Numbers are from 10^(d-1) to min(N, 10^d - 1)
    start = powers_of_10_val[d-1]
    end = min(N, powers_of_10_val[d] - 1)
    if start > N: # No numbers with d digits up to N
        C_d[d] = 0
    else:
        C_d[d] = end - start + 1

# Calculate S_val^(d_0) = sum of numbers with d_0 digits up to N
S_val = {}
inv_2 = inverse(2)
for d0 in range(1, D + 1):
    if C_d[d0] == 0:
        S_val[d0] = 0
        continue
    start = powers_of_10_val[d0 - 1]
    end = min(N, powers_of_10_val[d0] - 1)
    count = end - start + 1
    sum_val = ((start % MOD + end % MOD) % MOD * (count % MOD) % MOD * inv_2) % MOD
    S_val[d0] = sum_val

# Calculate coefficients of P(x) = Prod (1 + 10^d x)^C_d
# We need coefficients up to x^N for P(x)
poly_terms_list = [(powers_of_10_val[d], C_d[d]) for d in range(1, D + 1) if C_d[d] > 0]
P_coeffs = compute_P_coeffs(poly_terms_list)

# Ensure P_coeffs has size N+1 (indices 0 to N)
while len(P_coeffs) <= N:
    P_coeffs.append(0)
P_coeffs = P_coeffs[:N+1]

# Calculate T_d0_star for each d0
T_d0_star = {}
for d0 in range(1, D + 1):
    # Compute a_m^(d0) = [x^m] P(x) / (1 + 10^d0 x) using recurrence
    a_m_d0 = [0] * N # Need a_m^(d0) for m = 0 to N-1
    a_m_d0[0] = P_coeffs[0]
    pow10_d0_mod = powers_of_10_val[d0] % MOD
    for m in range(1, N):
        a_m_d0[m] = (P_coeffs[m] - (pow10_d0_mod * a_m_d0[m-1]) % MOD + MOD) % MOD

    # Compute T_d0_star = sum_{m=0}^{N-1} (N-m-1)! m! a_m^(d0)
    current_T_d0_star = 0
    for m in range(N): # m goes from 0 to N-1
        term = (fact[N-m-1] * fact[m]) % MOD
        term = (term * a_m_d0[m]) % MOD
        current_T_d0_star = (current_T_d0_star + term) % MOD
    T_d0_star[d0] = current_T_d0_star

# Final sum = sum_{d0=1}^D S_val^(d0) * T_d0_star[d0]
total_sum = 0
for d0 in range(1, D + 1):
    term = (S_val[d0] * T_d0_star[d0]) % MOD
    total_sum = (total_sum + term) % MOD

print(total_sum)