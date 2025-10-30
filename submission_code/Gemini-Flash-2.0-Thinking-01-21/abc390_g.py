import sys

# Modulo
MOD = 998244353

# NTT functions
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

def ntt(a, invert):
    n = len(a)
    j = 0
    for i in range(1, n):
        b = n >> 1
        while j & b:
            j ^= b
            b >>= 1
        j ^= b
        if i < j:
            a[i], a[j] = a[j], a[i]

    for len_ in range(2, n + 1, 2):
        half_len = len_ // 2
        wlen = power(3, (MOD - 1) // len_)
        if invert:
            wlen = inverse(wlen)

        for i in range(0, n, len_):
            w = 1
            for j in range(half_len):
                u = a[i + j]
                v = (a[i + half_len + j] * w) % MOD
                a[i + j] = (u + v) % MOD
                a[i + half_len + j] = (u - v + MOD) % MOD
                w = (w * wlen) % MOD

    if invert:
        inv_n = inverse(n)
        for i in range(n):
            a[i] = (a[i] * inv_n) % MOD

def multiply(poly1, poly2):
    size = len(poly1) + len(poly2) -1
    if size <= 0: return [0]
    n = 1
    while n < size:
        n <<= 1
    
    a = poly1[:] + [0] * (n - len(poly1))
    b = poly2[:] + [0] * (n - len(poly2))

    ntt(a, False)
    ntt(b, False)

    c = [(a[i] * b[i]) % MOD for i in range(n)]

    ntt(c, True)

    return c[:size] # Resulting polynomial has degree sum of degrees

# Precompute factorials
MAX_N = 200000
fact = [1] * (MAX_N + 1)
invFact = [1] * (MAX_N + 1)
for i in range(2, MAX_N + 1):
    fact[i] = (fact[i - 1] * i) % MOD
invFact[MAX_N] = inverse(fact[MAX_N])
for i in range(MAX_N - 1, 1, -1):
    invFact[i] = (invFact[i + 1] * (i + 1)) % MOD

def nCr_mod_p(n, r):
    if r < 0 or r > n:
        return 0
    return (((fact[n] * invFact[r]) % MOD) * invFact[n - r]) % MOD

# Adjusted powers of 10 precomputation limit
# Max exponent p*d is approx (N-1) * (log10(N)+1)
# For N=2e5, log10(N) approx 5.3. D=6. Max exponent approx 2e5 * 6 = 1.2e6
MAX_EXPONENT_10 = MAX_N * 7 
powers_of_10_vals = [1] * (MAX_EXPONENT_10 + 1)
for i in range(1, MAX_EXPONENT_10 + 1):
     powers_of_10_vals[i] = (powers_of_10_vals[i-1] * 10) % MOD

def power_of_10_mod(exp):
     if exp < 0: return inverse(power(10, -exp))
     if exp > MAX_EXPONENT_10: # Should not happen with correct bounds
          return power(10, exp) % MOD
     return powers_of_10_vals[exp]

# Main logic
def solve():
    N = int(sys.stdin.readline())

    if N == 0:
        print(0)
        return
    if N == 1:
        print(1)
        return

    # 1. Compute C_k coefficients
    # C_k_digits[d] = count of numbers in {1..N} with d digits
    MAX_DIGITS_N = len(str(N))
    C_k_digits = [0] * (MAX_DIGITS_N + 2) # Include possible N=10^k
    for d in range(1, MAX_DIGITS_N + 2):
        lower_bound = power(10, d - 1)
        upper_bound = power(10, d) - 1
        count = max(0, min(N, upper_bound) - lower_bound + 1)
        C_k_digits[d] = count
    
    # P_N(x) = prod (1 + 10^d x)^C_d
    # Use polynomial multiplication (NTT)
    current_poly = [1] # P_0(x) = 1

    for d in range(1, MAX_DIGITS_N + 2):
        count = C_k_digits[d]
        if count == 0:
            continue
        
        # Build polynomial (1 + 10^d x)^count
        # Use binary exponentiation for polynomials using NTT
        # Compute (1 + 10^d x)^count
        factor_poly_base = [1, power_of_10_mod(d)] # (1 + 10^d x)
        
        temp_poly = [1] # result starts with 1
        poly_power = count
        poly_base = factor_poly_base[:]

        while poly_power > 0:
             if poly_power % 2 == 1:
                 temp_poly = multiply(temp_poly, poly_base)
             poly_base = multiply(poly_base, poly_base)
             poly_power //= 2
        
        current_poly = multiply(current_poly, temp_poly)
        # Keep only coeffs up to x^N + degree of subsequent factors. Max degree is N.
        # The degree grows up to N. We need coeffs up to x^N for C_k.
        current_poly = current_poly[:min(N + 1, len(current_poly))]


    C = current_poly # C_k are coefficients of x^k in P_N(x)
    # Pad C with zeros if needed to size N+1
    C += [0] * (N + 1 - len(C))

    # 2. Compute U_p = sum_{j=1..N} j * (10^d(j))^p
    U = [0] * N # U_p for p = 0 to N-1
    sum_j_by_digits = [0] * (MAX_DIGITS_N + 2)
    for d in range(1, MAX_DIGITS_N + 2):
        lower_bound = power(10, d - 1)
        upper_bound = min(N, power(10, d) - 1)
        
        if lower_bound > upper_bound:
            continue
        
        # Sum of j from lower_bound to upper_bound
        # Arithmetic series: (first + last) * count / 2
        first = lower_bound % MOD
        last = upper_bound % MOD
        count = upper_bound - lower_bound + 1
        
        if count > 0:
            sum_val = (first + last) % MOD
            if sum_val % 2 == 0:
                 sum_val = (sum_val // 2) % MOD
            else:
                 sum_val = (sum_val * inverse(2)) % MOD
            sum_val = (sum_val * count) % MOD
            sum_j_by_digits[d] = sum_val

    for p in range(N):
        current_Up = 0
        for d in range(1, MAX_DIGITS_N + 2):
            pow_10_pd = power_of_10_mod(p * d) # exponent is p * d(j)
            current_Up = (current_Up + sum_j_by_digits[d] * pow_10_pd) % MOD
        U[p] = current_Up

    # 3. Compute T_p using convolution
    # T_p is coeff of x^(N-1-p) in A(x) * B_rev(x)
    # A(x) = sum_{k=0}^{N-1} C_k x^k
    # B_rev(x) = sum_{k=0}^{N-1} (k! * (N-1-k)!) x^k
    
    poly_A = C[:N] # C_0 to C_{N-1}
    poly_B_rev = [0] * N
    for k in range(N):
        poly_B_rev[k] = (fact[k] * fact[N - 1 - k]) % MOD

    poly_T_coeffs = multiply(poly_A, poly_B_rev)

    # T_p = poly_T_coeffs[N-1-p]
    T = [0] * N
    for p in range(N):
        if N - 1 - p < len(poly_T_coeffs):
             T[p] = poly_T_coeffs[N - 1 - p]
        else:
             T[p] = 0

    # 4. Compute total sum
    total_sum = 0
    for p in range(N):
        term = (T[p] * U[p]) % MOD
        if p % 2 == 1:
            total_sum = (total_sum - term + MOD) % MOD
        else:
            total_sum = (total_sum + term) % MOD

    print(total_sum)

solve()