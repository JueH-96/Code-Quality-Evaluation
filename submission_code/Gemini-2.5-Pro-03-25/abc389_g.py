# YOUR CODE HERE
import sys
sys.setrecursionlimit(3000) 

# MOD = 998244353 # Default NTT friendly modulus
# G = 3 # Default Primitive Root for MOD 998244353

MOD = 0 # Will be set in solve()
G = 5 # A common primitive root, works for many large primes like 998244353. Check if it works for the given P. It seems standard competitive programming problems use primes where G=3 or G=5 work.


# Fast exponentiation
def fast_power(base, power):
    """Computes base^power % MOD efficiently."""
    result = 1
    while power > 0:
        if power % 2 == 1:
            result = (result * base) % MOD
        base = (base * base) % MOD
        power //= 2
    return result

def inverse(a):
    """Computes the modular multiplicative inverse of a modulo MOD."""
    return fast_power(a, MOD - 2)

# NTT implementation
def ntt(a, inv):
    """Performs Number Theoretic Transform (NTT) in place."""
    n = len(a)
    j = 0
    # Bit-reversal permutation
    for i in range(1, n):
        rev = n >> 1
        while j >= rev:
            j -= rev
            rev >>= 1
        j += rev
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Cooley-Tukey butterfly updates
    step = 2
    while step <= n:
        half = step >> 1
        # Find primitive root for this step size
        w_step = pow(G, (MOD - 1) // step, MOD)
        if inv:
            w_step = inverse(w_step)
        
        for i in range(0, n, step):
            w = 1 # Current root of unity power
            for j in range(i, i + half):
                u = a[j]
                v = (w * a[j + half]) % MOD
                a[j] = (u + v) % MOD
                a[j + half] = (u - v + MOD) % MOD
                w = (w * w_step) % MOD
        step <<= 1
    
    # Scale by 1/n if inverse transform
    if inv:
        inv_n = inverse(n)
        for i in range(n):
            a[i] = (a[i] * inv_n) % MOD

def poly_mul_ntt(p1, p2, mod, max_deg):
    """Multiplies two polynomials p1 and p2 using NTT."""
    global MOD 
    MOD = mod # Set global MOD for NTT functions
    
    len1 = len(p1)
    len2 = len(p2)
    
    # Determine FFT size (power of 2)
    n = 1
    while n < len1 + len2 - 1:
        n <<= 1

    # Pad polynomials with zeros
    a = p1 + [0] * (n - len1)
    b = p2 + [0] * (n - len2)

    # Forward NTT
    ntt(a, False)
    ntt(b, False)

    # Pointwise multiplication
    c = [0] * n
    for i in range(n):
        c[i] = (a[i] * b[i]) % MOD

    # Inverse NTT
    ntt(c, True)
    
    # Truncate result to maximum degree
    res_len = min(len1 + len2 - 1, max_deg + 1)
    return c[:res_len]


def poly_add(p1, p2, mod):
    """Adds two polynomials p1 and p2."""
    max_len = max(len(p1), len(p2))
    res = [0] * max_len
    for i in range(max_len):
        val1 = p1[i] if i < len(p1) else 0
        val2 = p2[i] if i < len(p2) else 0
        res[i] = (val1 + val2) % mod
    return res

def poly_sub(p1, p2, mod):
    """Subtracts polynomial p2 from p1."""
    max_len = max(len(p1), len(p2))
    res = [0] * max_len
    for i in range(max_len):
        val1 = p1[i] if i < len(p1) else 0
        val2 = p2[i] if i < len(p2) else 0
        res[i] = (val1 - val2 + mod) % mod
    return res

def poly_scalar_mul(p, s, mod):
    """Multiplies polynomial p by a scalar s."""
    res = [0] * len(p)
    for i in range(len(p)):
        res[i] = (p[i] * s) % mod
    return res


def solve():
    N, P = map(int, sys.stdin.readline().split())
    
    MAX_EDGES = N * (N - 1) // 2
    global MOD
    MOD = P # Set global MOD used by NTT and other functions

    # Precompute factorials and their modular inverses
    MAX_FAC_COMB = N + 1 # For combinations nCr(N-1, k-1)
    MAX_FAC_EDGES = MAX_EDGES + 1 # For combinations nCr(edges, M)
    MAX_FAC = max(MAX_FAC_COMB, MAX_FAC_EDGES) # Overall maximum needed

    fact = [1] * (MAX_FAC + 1)
    inv_fact = [1] * (MAX_FAC + 1)
    if MAX_FAC >= 1:
      for i in range(1, MAX_FAC + 1):
          fact[i] = (fact[i-1] * i) % MOD

      inv_fact[MAX_FAC] = pow(fact[MAX_FAC], MOD - 2, MOD)
      for i in range(MAX_FAC - 1, -1, -1):
          inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    memo_nCr = {}
    def nCr_mod(n, r):
        """Computes nCr modulo MOD using precomputed factorials."""
        state = (n,r)
        if state in memo_nCr: return memo_nCr[state]

        if r < 0 or r > n:
            return 0
        if r == 0 or r == n:
             return 1
        if r > n // 2:
             r = n - r
        
        if n > MAX_FAC:
             # This case indicates an issue with MAX_FAC calculation or unexpected input.
             print(f"Error: nCr out of bounds n={n} MAX_FAC={MAX_FAC}") 
             return 0 

        num = fact[n]
        den = (inv_fact[r] * inv_fact[n-r]) % MOD
        res = (num * den) % MOD
        memo_nCr[state] = res
        return res

    # T_poly[k] stores the polynomial for total graphs on k vertices
    # Coefficient of x^M is the number of graphs with M edges. T(k, M) = nCr(k*(k-1)/2, M).
    T_poly = [[] for _ in range(N + 1)]
    T_poly[0] = [1] # Base case: 1 way for 0 vertices (empty graph), 0 edges.
    for k in range(1, N + 1):
        max_k_edges = k * (k - 1) // 2
        if max_k_edges < 0: max_k_edges = 0
        
        # Calculate maximum possible edges, limited by problem's overall MAX_EDGES
        current_poly_len = min(max_k_edges, MAX_EDGES) + 1
        current_poly = [0] * current_poly_len
        
        num_possible_edges = k * (k-1) // 2
        if num_possible_edges < 0: num_possible_edges = 0
        
        # Fill polynomial coefficients
        for m in range(current_poly_len):
             current_poly[m] = nCr_mod(num_possible_edges, m)
        T_poly[k] = current_poly

    # C_P_poly[k] stores the polynomial for connected graphs on k vertices satisfying property P.
    C_P_poly = [[] for _ in range(N + 1)]
    # Note: Property P: Connected graph with |V_even| = |V_odd|. This requires k to be even.
    
    # Base case: C_P(k, M) = 0 for k=1 (odd). C_P_poly[1] = []
    
    if N >= 2:
       # Base case N=2: One edge graph {1,2}. d(1,1)=0, d(1,2)=1. Ve={1}, Vo={2}. |Ve|=1, |Vo|=1. N=2, N/2=1. Condition holds.
       # C_P(2, 1)=1. C_P(2, M)=0 for M != 1.
       C_P_poly[2] = [0, 1] # Coefficients for M=0, M=1
       # Pad with zeros up to MAX_EDGES? Not necessary for empty lists approach.
    
    # DP calculation using the relation $C_P(N) = T_P(N) - \sum_{k=1}^{N-1} \binom{N-1}{k-1} C_P(k) T(N-k)$
    # where T_P(N) is the number of graphs on N vertices where the component containing vertex 1 satisfies property P.
    # $T_P(N) = \sum_{k=1}^{N} \binom{N-1}{k-1} C_P(k) T(N-k)$
    
    for k_curr in range(3, N + 1):
        
        # Property P is impossible for odd k, so C_P_poly is empty/zero polynomial
        if k_curr % 2 != 0: 
            continue
        
        # Compute T_P polynomial for k_curr vertices
        # T_P(N) = sum_{k=1..N} nCr(N-1, k-1) * C_P(k) * T(N-k)
        # This is polynomial T_P_curr_poly = sum_{k=1..k_curr} ...
        T_P_curr_poly = [0] * (MAX_EDGES + 1) # Initialize with zeros
        for k_prev in range(1, k_curr + 1): # Sum includes k_prev = k_curr
            comb = nCr_mod(k_curr - 1, k_prev - 1)
            if comb == 0: continue
            
            # Skip if C_P_poly[k_prev] is empty (or zero polynomial)
            if not C_P_poly[k_prev]: continue 
            
            # Polynomial multiplication part
            # Ensure T_poly[k_curr - k_prev] is available
            if k_curr - k_prev < 0: continue # Should not happen
            
            # Multiply C_P_poly[k_prev] * T_poly[k_curr - k_prev]
            prod_poly = poly_mul_ntt(C_P_poly[k_prev], T_poly[k_curr - k_prev], MOD, MAX_EDGES)
            
            # Scale by combination factor
            term_poly = poly_scalar_mul(prod_poly, comb, MOD)
            
            # Add to the total T_P polynomial
            T_P_curr_poly = poly_add(T_P_curr_poly, term_poly, MOD)

        # Compute the Sum term part for the main recurrence C_P = T_P - Sum
        # Sum = sum_{k=1..N-1} nCr(N-1, k-1) * C_P(k) * T(N-k)
        Sum_term_poly = [0] * (MAX_EDGES + 1) # Initialize
        for k_prev in range(1, k_curr): # Sum up to k_curr - 1
            comb = nCr_mod(k_curr - 1, k_prev - 1)
            if comb == 0: continue
            if not C_P_poly[k_prev]: continue 
            
            if k_curr - k_prev < 0: continue

            prod_poly = poly_mul_ntt(C_P_poly[k_prev], T_poly[k_curr - k_prev], MOD, MAX_EDGES)
            term_poly = poly_scalar_mul(prod_poly, comb, MOD)
            Sum_term_poly = poly_add(Sum_term_poly, term_poly, MOD)

        # Finally, compute C_P_poly[k_curr] using the relation
        # $C_P(N) = T_P(N) - \sum_{k=1}^{N-1} \binom{N-1}{k-1} C_P(k) T(N-k)$
        C_P_poly[k_curr] = poly_sub(T_P_curr_poly, Sum_term_poly, MOD)

    # Extract final answers from the polynomial C_P_poly[N]
    final_ans_poly = C_P_poly[N]
    ans = []
    # Iterate M from N-1 to MAX_EDGES
    for M in range(N - 1, MAX_EDGES + 1):
        if M < len(final_ans_poly):
             ans.append(final_ans_poly[M])
        else:
             # If degree M is not present, count is 0
             ans.append(0)

    # Print results separated by spaces
    print(*(ans))

solve()