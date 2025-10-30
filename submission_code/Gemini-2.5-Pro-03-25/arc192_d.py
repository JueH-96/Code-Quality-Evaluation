# YOUR CODE HERE
import sys
from collections import defaultdict

# Set higher recursion depth if needed, although this DP is iterative
# sys.setrecursionlimit(2000) 

MOD = 998244353

def fast_pow(base, power):
    """
    Compute (base^power) % MOD efficiently.
    Handles non-negative power.
    The base is reduced modulo MOD before exponentiation.
    """
    result = 1
    base %= MOD # Ensure base is within modulo range initially
    # Handle case where base is 0
    if base == 0: return 0 if power > 0 else 1

    while power > 0:
        if power % 2 == 1:
            result = (result * base) % MOD
        base = (base * base) % MOD
        power //= 2
    return result

def inverse(a):
    """
    Compute modular inverse of a modulo MOD using Fermat's Little Theorem.
    MOD must be prime and a != 0 mod MOD.
    The input `a` is reduced modulo MOD.
    """
    a %= MOD
    if a == 0:
        # This case should not happen if P_i >= 1
        raise ValueError("Cannot compute inverse of 0")
    return fast_pow(a, MOD - 2)

# Cache for prime factorizations to avoid recomputing
_prime_factorization_cache = {}

def get_prime_factorization(num):
    """
    Compute prime factorization of num. Use cache for efficiency.
    Returns a dictionary {prime: exponent}.
    """
    if num in _prime_factorization_cache:
        # Return a copy to prevent modification of cached dict
        return dict(_prime_factorization_cache[num]) 
    
    factors = defaultdict(int)
    d = 2
    temp_num = num
    while d * d <= temp_num:
        while temp_num % d == 0:
            factors[d] += 1
            temp_num //= d
        d += 1
    if temp_num > 1:
        factors[temp_num] += 1
    
    # Convert defaultdict to dict for cache consistency
    result_factors = dict(factors)
    _prime_factorization_cache[num] = result_factors
    return dict(result_factors) # Return a copy

def solve():
    N = int(sys.stdin.readline())
    # Problem constraints N >= 2, so no need to handle N=0 or N=1 explicitly.
    
    A_vals = list(map(int, sys.stdin.readline().split()))

    # Precompute factorizations and gather all primes involved across all A_i
    primes = set()
    A_factorizations = []
    for i in range(N - 1):
        # Check for A_i=0 case? Problem states A_i are positive integers.
        if A_vals[i] <= 0:
             raise ValueError("A_i must be positive integers")
        factors = get_prime_factorization(A_vals[i])
        A_factorizations.append(factors)
        for p in factors:
            primes.add(p)

    sorted_primes = sorted(list(primes))
    prime_to_idx = {p: i for i, p in enumerate(sorted_primes)}
    num_primes = len(sorted_primes)

    # Precompute pairs (P, Q) and their corresponding DP transition info for each A_i
    choices_per_A = []
    for i in range(N - 1): # 0-based index for A_vals list, corresponds to A_{i+1} in problem statement
        A_i = A_vals[i]
        factors = A_factorizations[i]
        
        current_choices = []
        
        distinct_primes_list = list(factors.keys())
        num_distinct_primes = len(distinct_primes_list)
        
        # Iterate through all 2^k subsets of prime factors to form P_i and Q_i
        # k = num_distinct_primes
        for k_mask in range(1 << num_distinct_primes):
            P_i = 1
            Q_i = 1
            
            # d_p_values stores the difference v_p(Q_i) - v_p(P_i) for each prime p in sorted_primes
            d_p_values = [0] * num_primes
            
            for bit_idx in range(num_distinct_primes):
                p = distinct_primes_list[bit_idx]
                exponent = factors[p]
                
                # Calculate integer powers
                p_pow_exp = p**exponent
                
                if (k_mask >> bit_idx) & 1:
                    # Assign p^exponent to Q_i
                    Q_i *= p_pow_exp 
                    if p in prime_to_idx:
                       d_p_values[prime_to_idx[p]] = exponent # d_p = v_p(Q) - v_p(P) = exponent - 0
                else:
                    # Assign p^exponent to P_i
                    P_i *= p_pow_exp
                    if p in prime_to_idx:
                       d_p_values[prime_to_idx[p]] = -exponent # d_p = v_p(Q) - v_p(P) = 0 - exponent
            
            # Calculate V_i = (Q_i / P_i)^(N - (i+1)) mod MOD. Note N-i in problem statement for 1-based index i
            # For 0-based index i, power is N - (i+1) = N - 1 - i.
            term_power = N - 1 - i

            # Compute (Q_i * P_i^{-1} mod MOD)^term_power mod MOD
            # Need modular inverse of P_i. P_i >= 1 guaranteed.
            term_base = (Q_i % MOD) * inverse(P_i) % MOD
            term_val = fast_pow(term_base, term_power)

            current_choices.append({'d_p_values': tuple(d_p_values), 'term_val': term_val})
        
        choices_per_A.append(current_choices)

    # DP state: Key is a tuple representing ((D_p, m_p) for p in sorted_primes)
    # Value is the accumulated weight W = sum of product V_j terms.
    
    # Initialize DP table for step 0 (before processing A_1)
    # State contains (D_p = 0, m_p = 0) for all primes.
    initial_state_key = tuple((0, 0) for _ in range(num_primes))
    dp = {initial_state_key: 1}

    # Iterate through DP steps from i = 0 to N-2 (processing A_1 to A_{N-1})
    for i in range(N - 1): 
        new_dp = defaultdict(int)
        
        # Iterate through current states and their weights
        for state_key, W in dp.items():
            if W == 0: continue # Skip states with zero weight

            # Process each possible choice for A_i = A_vals[i]
            for choice in choices_per_A[i]:
                d_p_vals = choice['d_p_values'] # Delta values for exponents for this choice
                term_val = choice['term_val']   # Multiplicative factor V_i for this choice
                
                next_state_list = []
                # Compute the next state based on current state and choice delta values
                for p_idx in range(num_primes):
                    Dp_curr, mp_curr = state_key[p_idx] # Current (D_p, m_p) for prime p
                    dp_val = d_p_vals[p_idx]           # Change d_p for this choice
                    
                    Dp_next = Dp_curr + dp_val         # New D_p value
                    mp_next = min(mp_curr, Dp_next)    # New minimum m_p value encountered so far
                    next_state_list.append((Dp_next, mp_next)) # Store as tuple for hashability
                
                # Convert list of state parts to tuple to use as dictionary key
                next_state_key = tuple(next_state_list)

                # Update the weight for the next state
                new_W = (W * term_val) % MOD
                new_dp[next_state_key] = (new_dp[next_state_key] + new_W) % MOD

        dp = new_dp # Update DP table for the next step

    # Final summation after processing all A_i values
    TotalSum = 0
    
    # Check MOD value for Fermat's Little Theorem exponent calculation
    # Exponent should be modulo (MOD - 1)
    if MOD == 2: 
      MOD_Minus_1 = 1 # Euler totient phi(2) = 1
    else:
      MOD_Minus_1 = MOD - 1 # Since MOD is prime, phi(MOD) = MOD - 1

    # Iterate through final states after N-1 steps
    for state_key, W in dp.items():
        if W == 0: continue
        
        # Calculate the T2 factor: product over primes p of p^(-N * mp_final)
        T2_factor = 1
        for p_idx in range(num_primes):
            p = sorted_primes[p_idx]
            _Dp_final, mp_final = state_key[p_idx] # Final D_p not needed, only m_p
            
            # mp_final stores min(0, D_p,2, ..., D_p,N)
            # We need p^(-N * mp_final) = p^(N * (-mp_final)). Since mp_final <= 0, exponent is >= 0.
            if mp_final < 0: 
                # Calculate exponent N * (-mp_final)
                exponent = N * (-mp_final) 
                
                # Check if base p is multiple of MOD. It won't be since p is prime and MOD > 1000 potentially.
                # If p is prime, p % MOD == 0 only if p == MOD. Check this edge case?
                # Max prime in A_i factors <= 1000. MOD = 998244353. So p != MOD.
                
                # Compute base^exponent mod MOD using Fermat's Little Theorem property
                if MOD_Minus_1 > 0: # Avoid modulo by zero if MOD = 1 (not possible here)
                     effective_exponent = exponent % MOD_Minus_1
                else: # Should not happen for MOD > 1
                     effective_exponent = 0 
                
                term = fast_pow(p, effective_exponent)
                
                T2_factor = (T2_factor * term) % MOD
        
        # Add contribution of this final state to the total sum
        TotalSum = (TotalSum + W * T2_factor) % MOD

    print(TotalSum)

solve()