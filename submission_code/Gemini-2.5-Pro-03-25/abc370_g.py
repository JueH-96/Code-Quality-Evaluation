# YOUR CODE HERE
import sys
import math

# Set higher recursion depth for safety. The required depth depends on N.
# For N=10^10, sqrt(N)=10^5. The number of distinct floor(N/k) is ~2*sqrt(N).
# Recursion involves iterating through primes and values, depth could be significant.
# Set a reasonably high limit, typical CP platforms allow up to ~10^6 stack depth.
# If Memory Limit Exceeded, this might be too high. If RecursionError, too low.
try:
    # Adjust recursion depth based on typical platform limits and problem needs.
    # 200,000 seems safe for stack size; Python's default is often ~1000.
    # Testing locally might reveal needed depth better. If issues arise, try adjusting.
     sys.setrecursionlimit(200000) 
except Exception as e: # Handle potential errors if setting recursion limit fails
    # print(f"Could not set recursion depth: {e}", file=sys.stderr)
    pass

def solve():
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353

    # Precompute factorials and their inverses
    # Need up to M + log2(N) + buffer. log2(10^10) is approx 33.2. Use 65 as buffer.
    MAX_M_PLUS_LOGN = M + 65 
    
    # Cap memory usage if M is extremely large, to stay within typical limits (e.g. 512MB)
    # The precomputation size is roughly MAX_M_PLUS_LOGN * 8 bytes/int * 2 arrays.
    # MAX_M_PLUS_LOGN=200000 -> ~3.2 MB. Should be fine.
    if MAX_M_PLUS_LOGN > 200000: 
        MAX_M_PLUS_LOGN = 200000 
    
    fact = [1] * (MAX_M_PLUS_LOGN + 1)
    inv_fact = [1] * (MAX_M_PLUS_LOGN + 1)
    for i in range(1, MAX_M_PLUS_LOGN + 1):
        fact[i] = (fact[i-1] * i) % MOD

    # Compute modular inverse of the largest factorial using Fermat's Little Theorem
    inv_fact[MAX_M_PLUS_LOGN] = pow(fact[MAX_M_PLUS_LOGN], MOD - 2, MOD)
    # Compute other inverse factorials iteratively
    for i in range(MAX_M_PLUS_LOGN - 1, -1, -1):
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    # Function to compute nCr mod P using precomputed factorials
    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        # Check if n and r are within precomputed range
        if n >= len(fact):
             # This indicates an error or misunderstanding of constraints.
             # Based on N <= 10^10 and M <= 10^5, n = e + M - 1 should not exceed ~100065.
             # print(f"Error: nCr request n={n} exceeds precomputed limit {len(fact)-1}", file=sys.stderr)
             return 0 # Return 0 safely if out of bounds

        # Compute nCr using modular arithmetic
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n-r]) % MOD
        return (num * den) % MOD

    # Setup F_P^2 arithmetic, representing elements as a + b*alpha where alpha^2 = -3 mod P
    alpha_sq = (MOD - 3) % MOD # P-3
    inv2 = pow(2, MOD - 2, MOD) # Modular inverse of 2
    
    # Define omega = (-1 + alpha)/2 in F_P^2
    omega_real = (MOD - 1) * inv2 % MOD # Real part: (P-1)/2
    omega_imag = inv2 # Imaginary part: 1/2
    omega = (omega_real, omega_imag) 
    
    # omega^2 = conjugate(omega) = (-1 - alpha)/2 in this field construction
    omega2 = (omega_real, (MOD - omega_imag) % MOD) # Real part same, imag part negated

    # Helper functions for F_P^2 arithmetic
    def fp2_add(x, y):
        # Add components modulo P
        return ((x[0] + y[0]) % MOD, (x[1] + y[1]) % MOD)

    def fp2_mul(x, y):
        # Multiply elements (a+b*alpha)(c+d*alpha) = (ac + bd*alpha^2) + (ad+bc)*alpha
        a, b = x
        c, d = y
        ac = (a * c) % MOD
        bd = (b * d) % MOD
        ad = (a * d) % MOD
        bc = (b * c) % MOD
        
        real = (ac + bd * alpha_sq) % MOD
        imag = (ad + bc) % MOD
        return (real, imag)

    # Function to compute g_k(p^e) = omega^{k * sigma(p^e)} * C(e+M-1, M-1)
    def calculate_g_k(p, e, k, M):
        # Calculate binomial coefficient C(e+M-1, M-1) mod P
        comb = nCr_mod(e + M - 1, M - 1)
        if comb == 0: return (0, 0) # If combination is 0, result is 0

        # Determine sigma(p^e) mod 3 based on p mod 3 and e
        term_val_mod3 = -1 # Placeholder
        if p == 3:
            term_val_mod3 = 1 # sigma(3^e) = 1 (mod 3) for e >= 0
        elif p % 3 == 1:
            term_val_mod3 = (e + 1) % 3 # sigma(p^e) = e+1 (mod 3)
        else: # p % 3 == 2
             term_val_mod3 = 1 if e % 2 == 0 else 0 # sigma(p^e) = 1 if e even, 0 if e odd (mod 3)

        # Calculate the exponent for omega: (k * sigma(p^e)) mod 3
        exponent = (k * term_val_mod3) % 3
        
        # Determine omega^exponent in F_P^2
        omega_pow_k_sigma = (1, 0) # Represents omega^0 = 1
        if exponent == 1:
            omega_pow_k_sigma = omega # Represents omega^1
        elif exponent == 2:
             omega_pow_k_sigma = omega2 # Represents omega^2

        # Return final result: product of combination (real number) and omega power (F_P^2 element)
        return fp2_mul((comb, 0), omega_pow_k_sigma)

    # Create wrapper functions for g_k(p^e) calculation. These capture M.
    # Used as arguments to the summation algorithm.
    def f0_pk_wrapper(p, k): return calculate_g_k(p, k, 0, M)
    def f1_pk_wrapper(p, k): return calculate_g_k(p, k, 1, M)
    def f2_pk_wrapper(p, k): return calculate_g_k(p, k, 2, M)

    # Implementation of summing a multiplicative function f(n) up to N.
    # Uses a recursive approach with memoization, similar to algorithms used for Project Euler problems
    # or competitive programming libraries (e.g. Min_25 sieve related techniques).
    
    primes_global = [] # Store primes up to sqrt(N) globally
    num_primes_global = 0
    memo_sum_func = {} # Memoization cache for S(n, k) results, shared across calls for S0, S1, S2

    def sum_multiplicative_function(target_N, func_pk_fp2_wrapper):
        nonlocal primes_global, num_primes_global, memo_sum_func
        
        sqrt_N = int(math.isqrt(target_N))
        
        # Ensure primes list is computed up to sqrt_N
        current_max_prime = primes_global[-1] if primes_global else 0
        if sqrt_N > current_max_prime:
             primes_global = []
             is_prime = [True] * (sqrt_N + 1)
             is_prime[0] = is_prime[1] = False
             for p in range(2, sqrt_N + 1):
                 if is_prime[p]:
                     primes_global.append(p)
                     for i in range(p * p, sqrt_N + 1, p): # Sieve multiples
                         is_prime[i] = False
             num_primes_global = len(primes_global)

        # Use function's identity for memoization key to distinguish S0, S1, S2 calls
        memo_key_prefix = id(func_pk_fp2_wrapper) 
        
        def S(n, k): # Computes Sum_{i=1..n, lpf(i) >= primes_global[k]} f(i)
            if n == 0: return (0,0) # Base case: sum up to 0 is 0
            
            # Base case: If smallest prime factor considered (p_k) is > n, 
            # the only number <= n satisfying lpf(i) >= p_k is 1. Return f(1).
            # Note: f(1) = g_k(1) = omega^k. Here, k=0,1,2. We calculate f(1) = 1 = (1,0).
            # The base case should return (1,0) representing f(1).
            if k >= num_primes_global or primes_global[k] > n:
                return (1,0) 
            
            state = (memo_key_prefix, n, k) # State includes function ID, n, and prime index k
            if state in memo_sum_func: return memo_sum_func[state] # Return cached result

            # Calculate S(n, k) using recursive formula:
            # S(n, k) = S(n, k+1) + Sum_{e>=1, p_k^e <= n} f(p_k^e) * S(n / p_k^e, k+1)
            
            # Contribution from numbers with lpf > p_k (handled by prime index k+1)
            res = S(n, k + 1)
            
            pk = primes_global[k] # Current prime p_k
            pk_power = pk # Start with p_k^1
            e = 1 # Exponent
            while True: 
                if pk_power > n: break # Stop if p_k^e exceeds n

                # Calculate f(p_k^e) using the provided function wrapper
                f_pk_e = func_pk_fp2_wrapper(pk, e)
                
                # Recursive call for S(n // pk_power, k + 1)
                # This sums f(m) where m <= n/pk^e and lpf(m) >= p_{k+1}
                term_val = S(n // pk_power, k + 1)
                
                # Add contribution: f(p_k^e) * S(...)
                res = fp2_add(res, fp2_mul(f_pk_e, term_val))
                
                # Check to prevent overflow before calculating next power p_k^{e+1}
                # If pk > n / pk_power, then pk * pk_power > n. Stop loop.
                if pk > n // pk_power: 
                     break

                pk_power *= pk # Move to next power p_k^{e+1}
                e += 1
                
            memo_sum_func[state] = res # Cache the computed result
            return res

        # The main call S(target_N, 0) computes the total sum Sum_{i=1..N} f(i)
        return S(target_N, 0)

    
    # Compute S0, S1, S2 using the summation function
    S0_fp2 = sum_multiplicative_function(N, f0_pk_wrapper)
    # It's safe to reuse the memoization cache if states are properly keyed with function ID
    S1_fp2 = sum_multiplicative_function(N, f1_pk_wrapper)
    S2_fp2 = sum_multiplicative_function(N, f2_pk_wrapper)
    
    # S0 should be a real number (integer mod P)
    S0 = S0_fp2[0] 
    S1 = S1_fp2
    S2 = S2_fp2
    
    # Combine results: Final Answer = (S0 + S1 + S2) / 3 mod P
    INV3 = pow(3, MOD - 2, MOD) # Modular inverse of 3
    
    # Sum S0, S1, S2 in F_P^2. S0 is represented as (S0, 0).
    TotalSum_FP2 = fp2_add((S0, 0), fp2_add(S1, S2))
    
    # The final result must be an integer mod P. We take the real part of the total sum.
    # This relies on the property S1 + S2 = 2 * Real(S1) which holds if S2 is conjugate of S1.
    # The formula works even if not conjugate, as the sum S0+S1+S2 is guaranteed to be real.
    FinalAnswer = (TotalSum_FP2[0] * INV3) % MOD
    
    print(FinalAnswer)

solve()