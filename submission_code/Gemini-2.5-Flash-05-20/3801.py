import functools

class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        
        # Precompute prime factorizations for digits 1-9
        # Only interested in primes 2, 3, 5, 7
        prime_factors_digits = {}
        for d in range(1, 10): # Digits 1-9
            temp_d = d
            factors = {2: 0, 3: 0, 5: 0, 7: 0}
            if temp_d % 2 == 0:
                while temp_d % 2 == 0:
                    factors[2] += 1
                    temp_d //= 2
            if temp_d % 3 == 0:
                while temp_d % 3 == 0:
                    factors[3] += 1
                    temp_d //= 3
            if temp_d % 5 == 0:
                while temp_d % 5 == 0:
                    factors[5] += 1
                    temp_d //= 5
            if temp_d % 7 == 0:
                while temp_d % 7 == 0:
                    factors[7] += 1
                    temp_d //= 7
            prime_factors_digits[d] = factors
        
        # Precompute prime factorizations for sums (1 to 81)
        # Also determine if a sum has a prime factor > 7
        prime_factors_sums = {}
        sum_has_large_prime_factor = [False] * 82 # Index 0 unused
        # Primes larger than 7 but <= 81
        large_primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
        
        for s_val in range(1, 82):
            temp_s = s_val
            factors = {2: 0, 3: 0, 5: 0, 7: 0}
            
            # Check for large prime factors
            for p in large_primes:
                if temp_s % p == 0:
                    sum_has_large_prime_factor[s_val] = True
                    break

            # Get factors for 2,3,5,7
            if temp_s % 2 == 0:
                while temp_s % 2 == 0:
                    factors[2] += 1
                    temp_s //= 2
            if temp_s % 3 == 0:
                while temp_s % 3 == 0:
                    factors[3] += 1
                    temp_s //= 3
            if temp_s % 5 == 0:
                while temp_s % 5 == 0:
                    factors[5] += 1
                    temp_s //= 5
            if temp_s % 7 == 0:
                while temp_s % 7 == 0:
                    factors[7] += 1
                    temp_s //= 7
            prime_factors_sums[s_val] = factors
        
        # Maximum exponents for primes 2,3,5,7 needed for product to satisfy any sum up to 81.
        # These are upper bounds for the `v_p(sum)` values.
        # We cap the `exp_p` in the DP state at `limit + 1` to signify "at least `limit`".
        exp_limit = {2: 6, 3: 4, 5: 2, 7: 2} # v_2(64)=6, v_3(81)=4, v_5(25)=2, v_7(49)=2
        
        # S will be the string representation of N for the current count(N) call
        S = "" 

        @functools.lru_cache(None)
        def dp(idx, current_sum, exp2, exp3, exp5, exp7, is_tight, is_started, has_significant_zero):
            # idx: current digit position
            # current_sum: sum of digits chosen so far (for the *actual* number)
            # exp_p: exponent of prime p in product of digits (for the *actual* number)
            # is_tight: boolean, true if current choice is restricted by S[idx]
            # is_started: boolean, true if we have placed any non-zero digit, meaning we started forming the actual number
            # has_significant_zero: boolean, true if we have placed any '0' digit *after* is_started became true

            if idx == len(S):
                if not is_started: # Formed an empty number or all leading zeros (e.g., "0", "00")
                    return 0
                
                # We have formed a valid positive integer
                if has_significant_zero: # Product of digits is 0
                    return 1 # 0 is divisible by any positive sum
                
                # No '0' digits, product > 0. Check divisibility.
                # If sum has any prime factor > 7, it's not beautiful
                if sum_has_large_prime_factor[current_sum]:
                    return 0 
                
                # Check prime factor counts for 2,3,5,7 against required factors from current_sum
                required_factors = prime_factors_sums[current_sum]
                
                if exp2 < required_factors[2]: return 0
                if exp3 < required_factors[3]: return 0
                if exp5 < required_factors[5]: return 0
                if exp7 < required_factors[7]: return 0
                
                return 1

            ans = 0
            upper_bound = int(S[idx]) if is_tight else 9

            for digit in range(upper_bound + 1):
                if not is_started and digit == 0:
                    # Still placing leading zeros. Current sum and product exponents don't change.
                    # 'has_significant_zero' remains False because this zero is not part of the 'actual' number.
                    ans += dp(idx + 1, current_sum, exp2, exp3, exp5, exp7, 
                              is_tight and (digit == upper_bound), False, False)
                else:
                    # This digit is part of the actual number (either first non-zero digit or after that).
                    new_sum = current_sum + digit
                    # Optimization: sum limit check. Max sum is 81.
                    if new_sum > 81: 
                        continue 
                    
                    new_exp2, new_exp3, new_exp5, new_exp7 = exp2, exp3, exp5, exp7
                    if digit > 0: # Update product exponents only for non-zero digits
                        digit_factors = prime_factors_digits[digit]
                        new_exp2 = min(exp_limit[2] + 1, exp2 + digit_factors[2]) # Cap for exp storage
                        new_exp3 = min(exp_limit[3] + 1, exp3 + digit_factors[3])
                        new_exp5 = min(exp_limit[5] + 1, exp5 + digit_factors[5])
                        new_exp7 = min(exp_limit[7] + 1, exp7 + digit_factors[7])
                        
                    new_has_significant_zero = has_significant_zero or (digit == 0) # This digit '0' is significant

                    ans += dp(idx + 1, new_sum, new_exp2, new_exp3, new_exp5, new_exp7,
                              is_tight and (digit == upper_bound), True, new_has_significant_zero)
            return ans

        def calculate_count_up_to_N(n: int) -> int:
            nonlocal S
            S = str(n)
            dp.cache_clear() # Clear memoization table for each call
            # Initial call: idx=0, sum=0, prod_exp all 0, tight=True, started=False, has_significant_zero=False
            return dp(0, 0, 0, 0, 0, 0, True, False, False)

        return calculate_count_up_to_N(r) - calculate_count_up_to_N(l - 1)