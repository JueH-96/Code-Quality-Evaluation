import sys

def read_int():
    return int(sys.stdin.readline())

MOD = 998244353
PRIMITIVE_ROOT = 3

def power(a, b):
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

def inverse(n):
    return power(n, MOD - 2)

# Precompute inverses for polynomial integration/logarithm
MAX_POLY_DEGREE = 200005 * 2 # N up to 2e5, needed up to 2N degrees
inv_list = [0, 1]
for i in range(2, MAX_POLY_DEGREE + 1):
    inv_list.append(MOD - (MOD // i) * inv_list[MOD % i] % MOD)

def ntt(a, inverse_ntt=False):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    for len_half in range(1, n // 2 + 1):
        len_full = 2 * len_half
        wlen = power(PRIMITIVE_ROOT, (MOD - 1) // len_full)
        if inverse_ntt:
            wlen = inverse(wlen)

        for i in range(0, n, len_full):
            w = 1
            for j in range(len_half):
                u = a[i + j]
                v = (a[i + j + len_half] * w) % MOD
                a[i + j] = (u + v) % MOD
                a[i + j + len_half] = (u - v + MOD) % MOD
                w = (w * wlen) % MOD

    if inverse_ntt:
        inv_n = inverse(n)
        for i in range(n):
            a[i] = (a[i] * inv_n) % MOD

    return a

def poly_mul(a, b):
    deg = len(a) + len(b) - 1
    n = 1
    while n < deg:
        n <<= 1
    
    a_padded = list(a) + [0] * (n - len(a))
    b_padded = list(b) + [0] * (n - len(b))

    a_padded = ntt(a_padded)
    b_padded = ntt(b_padded)

    c_padded = [(x * y) % MOD for x, y in zip(a_padded, b_padded)]
    c = ntt(c_padded, inverse_ntt=True)

    return c[:deg]

# Compute polynomial inverse A_inv(z) mod z^k
# A(z) * A_inv(z) = 1 (mod z^k)
def poly_inverse(a, k):
    n = 1
    while n < k:
        n <<= 1
    
    b = [inverse(a[0])] # Initial guess A_inv_0(z) mod z^1

    current_k = 1
    while current_k < k:
        current_k *= 2
        # Newton iteration: B_{i+1} = B_i * (2 - A * B_i) mod z^{2k}
        # A is a[:current_k]
        # B_i is b (current_k length)
        # A * B_i needs to be computed mod z^current_k.
        # Multiplication needs NTT size >= current_k * 2.
        
        tmp = poly_mul(list(a[:min(len(a), current_k)]), list(b[:min(len(b), current_k)]))
        tmp = tmp[:current_k] # Keep only terms up to z^{current_k-1}

        # 2 - A * B_i mod z^current_k
        for i in range(len(tmp)):
             tmp[i] = (MOD - tmp[i]) % MOD # - A * B_i
        tmp[0] = (tmp[0] + 2) % MOD # 2 - A * B_i

        # B_{i+1} = B_i * tmp mod z^current_k
        b_new = poly_mul(list(b[:min(len(b), current_k)]), tmp)
        b_new = b_new[:current_k]
        b = b_new # Update b
    
    return b[:k]

# Compute polynomial derivative B'(z)
def poly_derivative(b):
    if len(b) <= 1:
        return []
    return [(i + 1) * b[i + 1] % MOD for i in range(len(b) - 1)]

# Compute polynomial integral B_int(z)
# Integral of b_i z^i is b_i/(i+1) z^{i+1}
def poly_integral(b):
    res = [0] # Constant of integration is 0
    for i in range(len(b)):
        res.append(b[i] * inv_list[i + 1] % MOD)
    return res

# Compute polynomial exponentiation A(z) = exp(B(z)) mod z^k
# B(0) must be 0.
# A(z) = exp(B(z)). A'(z) = A(z) * B'(z).
# Newton iteration: A_{i+1} = A_i * (1 - log(A_i) + B) mod z^{2k}
def poly_exp(b, k):
    # Ensure B(0) is 0
    if len(b) > 0 and b[0] != 0:
        raise ValueError("B(0) must be 0 for poly_exp")
    
    n = 1
    while n < k:
        n <<= 1

    a = [1] # Initial guess exp(0) = 1 mod z^1
    
    current_k = 1
    while current_k < k:
        current_k *= 2
        # Need log(a) mod z^current_k
        # log(A) = Integral(A'/A) mod z^current_k
        
        a_deriv = poly_derivative(a[:min(len(a), current_k)])
        a_inv = poly_inverse(a[:min(len(a), current_k)], current_k)
        
        log_a_deriv = poly_mul(a_deriv, a_inv)
        log_a_deriv = log_a_deriv[:current_k -1] # Derivative reduces degree
        
        log_a = poly_integral(log_a_deriv)
        log_a = log_a[:current_k]

        # 1 - log(A_i) + B mod z^current_k
        # Ensure B is long enough
        b_padded = list(b[:min(len(b), current_k)]) + [0] * (current_k - len(b[:min(len(b), current_k)]))
        
        tmp = [0] * current_k
        for i in range(current_k):
             tmp[i] = (b_padded[i] - log_a[i] + MOD) % MOD # B - log(A_i)
        tmp[0] = (tmp[0] + 1) % MOD # 1 + B - log(A_i)

        # A_{i+1} = A_i * tmp mod z^current_k
        a_new = poly_mul(a[:min(len(a), current_k)], tmp)
        a_new = a_new[:current_k]
        a = a_new
    
    return a[:k]

# Compute polynomial P(z) = exp(Sum_{j=1}^k (-1)^{j-1} T_j/j z^j) mod z^k
# T = [T_1, T_2, ..., T_k]
def poly_from_log_series(T, k):
    b = [0] * (k + 1) # b_0 is 0
    
    for j in range(1, k + 1):
        term = T[j-1] * inv_list[j] % MOD
        if j % 2 == 0: # (-1)^{j-1} is negative
            term = (MOD - term) % MOD
        b[j] = term

    # Compute exp(B(z)) mod z^k. B(z) = sum_{i=0}^{k-1} b_i z^i
    # b has size k+1, b[:k] gives [b_0, ..., b_{k-1}]
    return poly_exp(b[:k], k)


def solve():
    N = read_int()

    # Precompute factorials and inverse factorials
    fact = [1] * (N + 1)
    for i in range(2, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    # 1. Compute C_p counts
    C = {}
    M = 0
    p10 = 1
    for p in range(1, 11): # Max digits for N <= 2e5 is 6. Go up to 10 just to be safe
        p10_next = p10 * 10
        # count is number of integers in [1, N] with p digits
        l = p10
        r = p10_next - 1
        count = max(0, min(N, r) - max(1, l) + 1)

        if count > 0:
            C[p] = count
            M = p
        if p10 > N:
            break
        p10 = p10_next

    # 2. Compute T_j = sum_{p=1}^M C_p (10^p)^j mod MOD for j=1..N
    T = [0] * N
    p10_mod = [0] * (M + 1)
    p10_mod[0] = 1
    for p in range(1, M + 1):
        p10_mod[p] = (p10_mod[p-1] * 10) % MOD

    for j in range(1, N + 1):
        current_Tj = 0
        for p in range(1, M + 1):
            if p in C:
                 term = C[p] * power(p10_mod[p], j) % MOD
                 current_Tj = (current_Tj + term) % MOD
        if j <= N:
             T[j-1] = current_Tj

    # 4. Compute E_0, ..., E_{N-1} using poly_exp
    # E_m are coefficients of P(z) = exp(B(z)) where B(z) = sum_{j=1}^N (-1)^{j-1} T_j/j z^j
    # poly_from_log_series computes exp(Sum_{j=1}^k b_j z^j) where b_j = (-1)^{j-1} T_j/j
    E = poly_from_log_series(T, N) # E = [E_0, E_1, ..., E_{N-1}]

    # 5. Compute T'_j = sum_{k=1}^N k (10^{d(k)})^j mod MOD for j=0..N-1
    T_prime = [0] * N
    inv2 = (MOD + 1) // 2
    
    sum_k_by_digits = {}
    p10 = 1
    for p in range(1, M + 1):
        p10_next = p10 * 10
        l = p10
        r = min(N, p10_next - 1)
        if l <= r:
            # sum is l + (l+1) + ... + r = (l+r)(r-l+1)/2
            count = r - l + 1
            s = (l % MOD + r % MOD) % MOD * (count % MOD) % MOD * inv2 % MOD
            sum_k_by_digits[p] = s
        p10 = p10_next

    for j in range(N):
        current_T_prime_j = 0
        for p in range(1, M + 1):
            if p in sum_k_by_digits:
                 term = sum_k_by_digits[p] * power(p10_mod[p], j) % MOD
                 current_T_prime_j = (current_T_prime_j + term) % MOD
        T_prime[j] = current_T_prime_j

    # 6. Compute V_m = sum_{j=0}^m (-1)^j E_{m-j} T'_j mod MOD for m=0..N-1
    # V_m = (E * B)_m where B_j = (-1)^j T'_j
    # E has size N (coeffs E_0...E_{N-1})
    # T_prime has size N (coeffs T'_0...T'_{N-1})
    B = [0] * N
    for j in range(N):
        B[j] = T_prime[j]
        if j % 2 != 0: # (-1)^j is -1
            B[j] = (MOD - B[j]) % MOD
            
    V = poly_mul(E, B)
    V = V[:N] # V_0 to V_{N-1}

    # 7. Compute total sum: sum_{m=0}^{N-1} (N-m-1)! m! V_m mod MOD
    total_sum = 0
    for m in range(N):
        term = fact[N - m - 1] * fact[m] % MOD * V[m] % MOD
        total_sum = (total_sum + term) % MOD

    print(total_sum)

solve()