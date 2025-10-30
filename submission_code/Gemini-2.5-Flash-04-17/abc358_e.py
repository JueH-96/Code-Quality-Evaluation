import sys

# Modulo
M = 998244353

# Primitive root (3 is a primitive root for 998244353)
G = 3

def power(a, b):
    """Calculates (a^b) % M"""
    res = 1
    a %= M
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % M
        a = (a * a) % M
        b //= 2
    return res

def inverse(a):
    """Calculates modular multiplicative inverse of a modulo M"""
    return power(a, M - 2) # Using Fermat's Little Theorem

def ntt(a, inverse_transform=False):
    """
    Performs Number Theoretic Transform (or inverse NTT) on a list of coefficients.
    The length of list `a` must be a power of 2.
    Modulo M and primitive root G are used from the global scope.
    """
    n = len(a)
    
    # Bit-reversal permutation
    j = 0
    for i in range(n):
        if i < j:
            a[i], a[j] = a[j], a[i]
        k = n // 2
        while k <= j:
            j -= k
            k //= 2
        j += k

    # Cooley-Tukey iterative step
    # `len_` is the size of the subproblems being merged
    for len_ in range(2, n + 1, 2):
        half_len = len_ // 2
        # Root of unity for this size
        wlen = power(G, (M - 1) // len_)
        if inverse_transform:
            wlen = inverse(wlen)

        # Iterate through the blocks of current size
        for i in range(0, n, len_):
            w = 1 # Current root of unity
            # Combine subproblems within the block
            for j in range(half_len):
                u = a[i + j]
                v = (a[i + j + half_len] * w) % M
                
                a[i + j] = (u + v) % M
                a[i + j + half_len] = (u - v + M) % M # Ensure result is non-negative
                
                w = (w * wlen) % M

    if inverse_transform:
        # Scale results by N^(-1)
        inv_n = inverse(n)
        for i in range(n):
            a[i] = (a[i] * inv_n) % M

# --- Main Logic ---
if __name__ == "__main__":
    K = int(sys.stdin.readline())
    C = list(map(int, sys.stdin.readline().split()))

    # Precompute factorials and inverse factorials modulo M up to K
    fact = [1] * (K + 1)
    inv_fact = [1] * (K + 1)
    for i in range(1, K + 1):
        fact[i] = (fact[i - 1] * i) % M

    # inv_fact[K] = (K!)^(-1) mod M
    inv_fact[K] = inverse(fact[K])
    # Calculate inverse factorials downwards
    for i in range(K - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % M

    # Determine minimum NTT length N
    # We are multiplying a polynomial of degree K by a polynomial of degree <= K.
    # To get coefficients up to degree K in the product, we need convolution up to index K.
    # A convolution of sequences of length K+1 and M_2+1 needs length >= (K+1) + (M_2+1) - 1
    # Here M_2 = min(C_i, K) <= K. So we need length >= (K+1) + (K+1) - 1 = 2K + 1.
    # NTT length N must be a power of 2 >= 2K + 1.
    N = 1
    while N < 2 * K + 1:
        N <<= 1 # N = N * 2

    # Coefficients of the product polynomial G(x) = sum(g_L * x^L / L!). Let's track g_L/L! as q_coeffs[L].
    # The coefficient of x^L in G(x) is g_L / L!
    # So q_coeffs[L] = g_L / L!
    # Initially, G(x) = 1 (representing the empty string of length 0).
    # G(x) = 1 + 0*x + 0*x^2 + ...
    # g_0 = 1, g_L = 0 for L > 0.
    # q_coeffs[0] = g_0 / 0! = 1 / 1 = 1.
    # q_coeffs[L] = g_L / L! = 0 / L! = 0 for L > 0.
    q_coeffs = [0] * (K + 1)
    q_coeffs[0] = 1

    # Iterate through each character type (A-Z)
    for c_limit in C:
        # Construct the polynomial for the current character type i.
        # P_i(x) = sum_{j=0}^{min(c_limit, K)} (x^j / j!)
        # The coefficients of P_i(x) are [1/0!, 1/1!, ..., 1/min(c_limit, K)!]
        p_coeffs = [0] * (K + 1)
        for j in range(min(c_limit, K) + 1):
            p_coeffs[j] = inv_fact[j]
            
        # Pad q_coeffs and p_coeffs to NTT length N
        q_padded = q_coeffs + [0] * (N - (K + 1))
        p_padded = p_coeffs + [0] * (N - (K + 1))

        # Compute NTT of both padded polynomials
        ntt(q_padded, False)
        ntt(p_padded, False)

        # Pointwise multiplication in the frequency domain
        # (A * B)(x) = sum_L (sum_{j=0}^L A_j B_{L-j}) x^L
        # Here, A_j = q_coeffs_old[j] = (old g_j) / j!
        # B_j = p_coeffs[j] = 1 / j! (for current char)
        # New q_coeffs_new[L] = (New g_L) / L!
        # The coefficient of x^L in the product is sum_{j=0}^L q_coeffs_old[j] * p_coeffs[L-j]
        # = sum_{j=0}^L ((old g_j) / j!) * (1 / (L-j)!)
        # This sum is equal to (New g_L) / L! * sum_{j=0}^L (L! / (j! * (L-j)!)) * (old g_j) / (L! / (j! * (L-j)!)) * 1
        # = (New g_L) / L! * sum_{j=0}^L binomial(L, j) * (old g_j) / L!
        # This is not the standard convolution structure (New g_L) / L! = sum_{j=0}^L (old g_j)/j! * 1/(L-j)!
        # The coefficient of x^L in the product $G^{(i)}(x) = G^{(i-1)}(x) \cdot P_i(x)$ is given by
        # $g_L^{(i)} / L! = \sum_{j=0}^L (g_j^{(i-1)} / j!) \cdot (1 / (L-j)!)$
        # This is indeed the convolution of sequences $(g_0^{(i-1)}/0!, g_1^{(i-1)}/1!, \dots)$ and $(1/0!, 1/1!, \dots)$.
        # So q_coeffs[L] correctly stores $g_L/L!$.

        r_padded = [(q_padded[j] * p_padded[j]) % M for j in range(N)]

        # Compute inverse NTT to get coefficients of the product polynomial
        ntt(r_padded, True)

        # The first K+1 coefficients are the new q_coeffs for the next iteration
        # These are coefficients of x^0, x^1, ..., x^K in the product polynomial G(x)
        q_coeffs = r_padded[0 : K + 1]

    # The final q_coeffs[L] is the coefficient of x^L in G(x), which is g_L / L!
    # The number of strings of length L is g_L = L! * (g_L / L!) = fact[L] * q_coeffs[L]
    # We need the total number of strings of length 1 to K

    total_strings = 0
    for L in range(1, K + 1):
        # Number of strings of length L is fact[L] * q_coeffs[L] modulo M
        num_strings_L = (fact[L] * q_coeffs[L]) % M
        total_strings = (total_strings + num_strings_L) % M

    # Print the final answer
    print(total_strings)