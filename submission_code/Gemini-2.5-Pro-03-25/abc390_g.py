# YOUR CODE HERE
import sys

# Set higher recursion depth if needed, although iterative NTT is used.
# sys.setrecursionlimit(2000) 

def solve():
    N = int(sys.stdin.readline())
    MOD = 998244353

    # Precompute factorials modulo MOD
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    # Compute num_digits[j] = number of digits of j
    # and ten_pow_digits[j] = 10^(num_digits[j]) mod MOD
    num_digits = [0] * (N + 1)
    ten_pow_digits = [0] * (N + 1)
    
    # Compute powers of 10 once per distinct digit length for efficiency
    pow10_vals = {}
    curr_pow_10 = 1
    curr_digits = 0 
    for i in range(1, N + 1):
      # Handle i=1 specifically to initialize
      if i == 1:
            curr_digits = 1
            curr_pow_10 = 1
      # Update digits length when i reaches the next power of 10
      # Check if i equals the current power of 10 * 10 (e.g., 10, 100, 1000...)
      elif i == curr_pow_10 * 10:
         curr_pow_10 *= 10
         curr_digits += 1
          
      num_digits[i] = curr_digits
      # Cache the computed power of 10 for this digit length
      if curr_digits not in pow10_vals:
           pow10_vals[curr_digits] = pow(10, curr_digits, MOD)
      ten_pow_digits[i] = pow10_vals[curr_digits]

    # Group numbers j by their number of digits d.
    # Compute S_d = sum of j such that num_digits[j] = d, modulo MOD.
    max_digits = num_digits[N]
    S_d = [0] * (max_digits + 1)
    
    # Modular inverse of 2 is needed for arithmetic series sum formula
    inv2 = pow(2, MOD - 2, MOD) 

    curr_pow_10 = 1
    start_idx = 1 # Tracks the beginning of the current range of numbers
    for d in range(1, max_digits + 1):
        # Determine the range of numbers with d digits: [10^(d-1), 10^d - 1]
        # Intersect this with the overall range [1, N]
        lower_bound = curr_pow_10
        upper_bound = min(N, curr_pow_10 * 10 - 1)
        
        # The actual range for this digit length within [1, N]
        current_start = max(start_idx, lower_bound) # Should always be lower_bound if start_idx is managed correctly
        current_end = upper_bound
        
        count = current_end - current_start + 1

        if count > 0:
          # Sum of arithmetic series: (count * (first + last) / 2) % MOD
          current_sum = (count % MOD * (current_start + current_end) % MOD * inv2) % MOD
          S_d[d] = current_sum
        
        # Update start_idx for the next potential digit group
        # Not needed with current logic using curr_pow_10
        curr_pow_10 *= 10
        if curr_pow_10 > N: # Optimization: If the next power of 10 exceeds N, we are done.
             break

    # P_d = 10^d mod M. These are already computed in pow10_vals.
    P_d = [0] * (max_digits + 1)
    P_d[0] = 1
    for d in range(1, max_digits + 1):
        P_d[d] = pow10_vals[d] 


    # Compute coefficients h_k = (-1)^k * sum_{d=1}^D S_d * (P_d)^k mod M for k = 0..N-1
    # This is part of the calculation related to polynomial H(y)
    h = [0] * N
    
    # current_Pd_pow_k[d] stores P_d^k for the current k
    current_Pd_pow_k = [1] * (max_digits + 1) 
    
    # Compute h_0 (k=0)
    term_sum = 0
    for d in range(1, max_digits + 1):
        term_sum = (term_sum + S_d[d] * current_Pd_pow_k[d]) % MOD
    h[0] = term_sum 
    
    # Compute h_k for k=1..N-1 iteratively
    neg_one_pow_k = 1 # Tracks (-1)^k
    for k in range(1, N):
        # Update (-1)^k efficiently using modular arithmetic
        neg_one_pow_k = (MOD - neg_one_pow_k) % MOD 
        term_sum = 0
        for d in range(1, max_digits + 1):
            # Update P_d^k to P_d^(k+1) for next iteration
            current_Pd_pow_k[d] = (current_Pd_pow_k[d] * P_d[d]) % MOD
            term_sum = (term_sum + S_d[d] * current_Pd_pow_k[d]) % MOD
        
        h[k] = (neg_one_pow_k * term_sum) % MOD

    # Number Theoretic Transform (NTT) implementation
    def fast_pow(base, power):
        """Computes base^power % MOD efficiently."""
        result = 1
        while power > 0:
            if power % 2 == 1:
                result = (result * base) % MOD
            base = (base * base) % MOD
            power //= 2
        return result

    def inverse(a):
        """Computes modular multiplicative inverse of a modulo MOD."""
        # Assumes MOD is prime ( Fermat's Little Theorem: a^(MOD-2) % MOD )
        return fast_pow(a, MOD - 2)

    primitive_root = 3 # Primitive root for MOD 998244353
    
    # Determine required length for NTT (must be power of 2)
    # Need length >= degree of resulting polynomial + 1
    # Product P_S * H has degree (N) + (N-1) = 2N-1. Need length >= 2N.
    MAX_LEN_NTT = 1
    while MAX_LEN_NTT <= 2 * N: 
        MAX_LEN_NTT <<= 1

    # Precompute roots of unity and their inverses
    roots = [1] * MAX_LEN_NTT
    inv_roots = [1] * MAX_LEN_NTT
    
    root = fast_pow(primitive_root, (MOD - 1) // MAX_LEN_NTT)
    inv_root = inverse(root)

    for i in range(1, MAX_LEN_NTT):
        roots[i] = (roots[i-1] * root) % MOD
    # Compute inverse roots based on forward roots
    # inv_roots[0] = 1 is already set
    inv_roots[1] = inv_root
    for i in range(2, MAX_LEN_NTT):
         inv_roots[i] = (inv_roots[i-1] * inv_root) % MOD

    # Precompute bit reversal permutation indices
    rev = [0] * MAX_LEN_NTT
    L = MAX_LEN_NTT.bit_length() - 1 # L = log2(MAX_LEN_NTT)
    for i in range(MAX_LEN_NTT):
         rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (L - 1))

    def ntt(a, length, inv):
        """Performs NTT in-place on list `a` of length `length`."""
        # Pad list `a` with zeros if it's shorter than `length`
        if len(a) < length:
            a.extend([0] * (length - len(a)))
        
        # Apply bit reversal permutation
        for i in range(length):
            if i < rev[i]:
                 a[i], a[rev[i]] = a[rev[i]], a[i]

        # Cooley-Tukey butterfly operations
        k = 1 # Current block size / 2
        while k < length:
            current_roots = inv_roots if inv else roots
            # step size in precomputed roots array depends on MAX_LEN_NTT
            step = MAX_LEN_NTT // (2*k) 
            
            for i in range(0, length, 2*k): # Iterate through blocks
                for j in range(k): # Iterate within block
                    # w is the (j*step)-th root of unity for length MAX_LEN_NTT
                    w = current_roots[j * step]
                    u = a[i + j]
                    v = (w * a[i + j + k]) % MOD
                    a[i + j] = (u + v) % MOD
                    a[i + j + k] = (u - v + MOD) % MOD # Use + MOD to ensure positive result
            k *= 2
        
        # Scale by 1/length for inverse transform
        if inv:
            inv_len = inverse(length)
            for i in range(length):
                a[i] = (a[i] * inv_len) % MOD

    def poly_multiply(p1, p2):
        """Multiplies two polynomials p1 and p2 using NTT."""
        len1 = len(p1)
        len2 = len(p2)
        
        # Determine resulting polynomial degree and NTT length needed
        res_len = len1 + len2 - 1
        ntt_len = 1
        while ntt_len < res_len:
            ntt_len <<= 1
        
        # Create copies to work with, padded to ntt_len
        a = list(p1) 
        b = list(p2)

        # Perform NTT on both polynomials
        ntt(a, ntt_len, False)
        ntt(b, ntt_len, False)

        # Pointwise multiplication in frequency domain
        res = [0] * ntt_len
        for i in range(ntt_len):
            res[i] = (a[i] * b[i]) % MOD

        # Inverse NTT to get coefficients of the product polynomial
        ntt(res, ntt_len, True)
        
        # Return the result, truncated to the actual degree
        return res[:res_len]

    
    def multiply_factors(factors):
        """Multiplies a list of polynomials using divide and conquer with NTT."""
        q = list(factors) # Use a copy to not modify the original list
        
        if not q: return [1] # Product of empty set is identity polynomial 1
        
        # Repeatedly pair up and multiply polynomials until only one remains
        while len(q) > 1:
            new_q = []
            for i in range(0, len(q), 2):
                if i + 1 < len(q): # If there's a pair
                    p1 = q[i]
                    p2 = q[i+1]
                    new_q.append(poly_multiply(p1, p2))
                else: # Odd number of polynomials, last one carries over
                    new_q.append(q[i])
            q = new_q
        return q[0] # The final resulting polynomial

    # Define the linear factors (1 + y * 10^{L_x}) for the polynomial P_S(y)
    linear_factors = []
    for x in range(1, N + 1):
         # Polynomial (1 + C*y) has coefficients [1, C]
         linear_factors.append([1, ten_pow_digits[x]])

    # Compute coefficients W(S, k) of P_S(y) = product of linear factors
    P_S_coeffs = multiply_factors(linear_factors) 

    # P_S polynomial has degree N. Coefficients W(S,0) .. W(S,N). Expected length N+1.
    # Ensure P_S_coeffs has length N+1 by padding/truncating if necessary
    if len(P_S_coeffs) < N + 1:
      P_S_coeffs.extend([0] * (N + 1 - len(P_S_coeffs)))
    P_S_coeffs = P_S_coeffs[:N+1] 

    # H(y) polynomial coefficients are h_0, ..., h_{N-1}. Length N.
    H_coeffs = h 

    # Compute G_k = coefficients of the product polynomial P_S(y) * H(y)
    G_poly_coeffs = poly_multiply(P_S_coeffs, H_coeffs)

    # Compute the final sum: S = sum_{k=0}^{N-1} (N-k-1)! * k! * G_k mod M
    total_sum = 0
    for k in range(N):
        # Calculate ((N-k-1)! * k!) % MOD
        term_factorials = (fact[N - k - 1] * fact[k]) % MOD
        
        # G_k is the k-th coefficient of the product polynomial G(y) = P_S(y)H(y)
        # Check boundary: G_poly_coeffs length is degree+1 = (2N-1)+1 = 2N
        if k < len(G_poly_coeffs):
             Gk = G_poly_coeffs[k]
             term = (term_factorials * Gk) % MOD
             total_sum = (total_sum + term) % MOD
        # If k >= len(G_poly_coeffs), Gk is effectively 0. Should not happen for k < N.

    print(total_sum)

solve()