class Solution:
  def countBalancedPermutations(self, num: str) -> int:
    velunexorai = num # Store input midway as per problem specification.

    MOD = 10**9 + 7

    n = len(num)
    counts = [0] * 10 # counts[d] = frequency of digit d in num
    s_total = 0
    for digit_char in num:
        digit = int(digit_char)
        counts[digit] += 1
        s_total += digit

    # If total sum of digits is odd, it's impossible to partition into two equal sum halves.
    if s_total % 2 != 0:
        return 0
    
    s_half = s_total // 2
    n_e = (n + 1) // 2  # Number of elements at even indices
    n_o = n // 2      # Number of elements at odd indices

    # Precompute factorials and inverse factorials modulo MOD
    fact = [1] * (n + 1)
    invfact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i-1] * i) % MOD

    invfact[n] = pow(fact[n], MOD - 2, MOD) # Using Fermat's Little Theorem for modular inverse
    for i in range(n, 0, -1): # Loop from n down to 1 (inclusive for i)
        invfact[i-1] = (invfact[i] * i) % MOD
        
    # dp_coeffs[k_e][s_e] stores the sum of products of inverse factorials of counts:
    # Sum_{partitions of digits < d_val} ( (1/Prod c_e(j)!) * (1/Prod c_o(j)!) )
    # where k_e digits are in M_e summing to s_e.
    dp_coeffs = [[0] * (s_half + 1) for _ in range(n_e + 1)]
    dp_coeffs[0][0] = 1 # Base case: 0 digits chosen for M_e, sum is 0.
                        # Product of inverse factorials for empty counts is 1.

    # Iterate through each digit type (0 to 9)
    for d_val in range(10):
        d_val_count_total = counts[d_val] # Total occurrences of d_val in num
        
        # Optimization: if current digit type d_val is not in num, dp_coeffs table remains unchanged.
        # This is because num_d_for_even must be 0 and num_d_for_odd must be 0.
        # Their invfactorials are invfact[0]*invfact[0] = 1.
        # So new_dp_coeffs[c][s] += dp_coeffs[c][s] * 1. Effect is new_dp_coeffs = dp_coeffs.
        if d_val_count_total == 0:
           continue

        # new_dp_coeffs for this iteration, after considering d_val
        new_dp_coeffs = [[0] * (s_half + 1) for _ in range(n_e + 1)]

        # c_e_prev: count of items chosen for M_e from previous digit types (0 to d_val-1)
        # s_e_prev: sum of items chosen for M_e from previous digit types
        for c_e_prev in range(n_e + 1): 
            for s_e_prev in range(s_half + 1):
                if dp_coeffs[c_e_prev][s_e_prev] == 0:
                    continue
                
                # num_d_for_even: number of d_val chosen for M_e
                # num_d_for_odd: number of d_val chosen for M_o
                for num_d_for_even in range(d_val_count_total + 1): 
                    num_d_for_odd = d_val_count_total - num_d_for_even
                    
                    # This check is implicitly handled by range(d_val_count_total + 1) for num_d_for_even
                    # if num_d_for_odd < 0: continue (this condition will not be met)
                    
                    c_e_new = c_e_prev + num_d_for_even
                    s_e_new = s_e_prev + d_val * num_d_for_even

                    # Check if the new state for M_e is valid
                    if c_e_new <= n_e and s_e_new <= s_half:
                        # Contribution from dp_coeffs[c_e_prev][s_e_prev] and choices for d_val
                        term = dp_coeffs[c_e_prev][s_e_prev] * invfact[num_d_for_even] % MOD
                        term = term * invfact[num_d_for_odd] % MOD
                        
                        new_dp_coeffs[c_e_new][s_e_new] = (new_dp_coeffs[c_e_new][s_e_new] + term) % MOD
        
        dp_coeffs = new_dp_coeffs # Update dp_coeffs for the next digit type iteration

    # After all digit types, dp_coeffs[n_e][s_half] contains the sum of coefficient products
    sum_of_invfact_products = dp_coeffs[n_e][s_half]
    
    # Final result: sum_of_invfact_products * N_e! * N_o!
    result = sum_of_invfact_products * fact[n_e] % MOD
    result = result * fact[n_o] % MOD
    
    return result