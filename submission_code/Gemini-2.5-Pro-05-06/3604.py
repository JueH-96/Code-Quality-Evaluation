class Solution:
  def numberOfWays(self, n: int, x: int, y: int) -> int:
    MOD = 10**9 + 7

    # Constraints: 1 <= n, x, y <= 1000.
    # max_fact_val is the maximum argument for N in C(N,R).
    # C(x, k) needs N up to x.
    # C(k, j) needs N up to k, where k <= min(n,x).
    # So, max N is x. Max x is 1000.
    max_fact_val = 1000 
    
    fact = [1] * (max_fact_val + 1)
    inv_fact = [1] * (max_fact_val + 1)

    for i in range(1, max_fact_val + 1):
        fact[i] = (fact[i-1] * i) % MOD

    # Fermat's Little Theorem for modular inverse: a^(MOD-2) % MOD
    inv_fact[max_fact_val] = pow(fact[max_fact_val], MOD - 2, MOD)
    # Compute inverse factorials iteratively: (i!)^-1 = ((i+1)!)^-1 * (i+1)
    for i in range(max_fact_val - 1, -1, -1): 
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD
        
    # Binomial coefficient C(num, r) % MOD
    def nCr_mod(num, r):
        if r < 0 or r > num:
            return 0
        # C(N,R) = N! / (R! * (N-R)!)
        numerator = fact[num]
        denominator = (inv_fact[r] * inv_fact[num-r]) % MOD
        return (numerator * denominator) % MOD

    total_ways = 0
    
    # k_limit is the maximum number of non-empty stages (bands)
    # k must be at least 1 (since n >= 1 performers guarantee at least one band)
    # k <= n (cannot have more bands than performers)
    # k <= x (cannot use more stages than available)
    k_limit = min(n, x)

    # Precompute j^n % MOD for j from 0 to k_limit.
    # j_pow_n[j_idx] stores j_idx^n % MOD.
    j_pow_n = [0] * (k_limit + 1)
    for j_idx in range(k_limit + 1):
        # pow(0, n, MOD) is 0 for n >= 1. pow(0,0,MOD) is 1 by Python's convention.
        # Since n >= 1 as per constraints, j_pow_n[0] = pow(0, n, MOD) = 0.
        j_pow_n[j_idx] = pow(j_idx, n, MOD)
        
    # Precompute y^k_val % MOD for k_val from 0 to k_limit.
    # y_pow_k_arr[k_idx] stores y^k_idx % MOD.
    y_pow_k_arr = [0] * (k_limit + 1)
    # Constraints: 1 <= y <= 1000.
    for k_idx in range(k_limit + 1): 
        y_pow_k_arr[k_idx] = pow(y, k_idx, MOD)


    # Sum over k_val (number of non-empty stages/bands) from 1 to k_limit.
    for k_val in range(1, k_limit + 1):
        # Calculate Stirling term: k_val! * S2(n, k_val)
        # This equals sum_{j_inner=0}^{k_val} [ (-1)^(k_val-j_inner) * C(k_val, j_inner) * j_inner^n ]
        # Since n >= 1, j_inner^n = 0 for j_inner=0. 
        # So, the term for j_inner=0 is C(k_val,0) * 0^n = 1 * 0 = 0.
        # The sum effectively starts from j_inner=1. The loop range [0, k_val] is fine.
        
        stirling_term_times_fact_k = 0
        for j_inner in range(k_val + 1): 
            term_j_contrib = (nCr_mod(k_val, j_inner) * j_pow_n[j_inner]) % MOD
            
            if (k_val - j_inner) % 2 == 1: # If (k_val - j_inner) is odd, sign is -1
                stirling_term_times_fact_k = (stirling_term_times_fact_k - term_j_contrib + MOD) % MOD
            else: # If (k_val - j_inner) is even, sign is +1
                stirling_term_times_fact_k = (stirling_term_times_fact_k + term_j_contrib) % MOD
        
        # Ways to choose k_val stages out of x: C(x, k_val)
        ways_to_choose_stages = nCr_mod(x, k_val)
        
        # Ways to assign scores to k_val bands: y^k_val
        ways_to_assign_scores = y_pow_k_arr[k_val]
        
        # Combine for this k_val: C(x, k_val) * (k_val! * S2(n, k_val)) * y^k_val
        term_for_this_k = (ways_to_choose_stages * stirling_term_times_fact_k) % MOD
        term_for_this_k = (term_for_this_k * ways_to_assign_scores) % MOD
        
        total_ways = (total_ways + term_for_this_k) % MOD
            
    return total_ways