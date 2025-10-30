class Solution:
  def minimumPossibleSum(self, n: int, target: int) -> int:
    MOD = 10**9 + 7

    # k0 represents the largest number x such that all integers from 1 to x
    # can be part of the beautiful array without any pair summing to target.
    # These are numbers 1, 2, ..., floor(target/2).
    k0 = target // 2
    
    # m is the count of numbers chosen from the range [1, k0].
    # If n is small enough, we only pick numbers from this range.
    # Otherwise, we pick all k0 numbers from this range.
    m = min(n, k0)

    sum1 = 0
    if m > 0:
        # Sum of first m numbers: 1, 2, ..., m.
        # Formula: m * (m + 1) / 2
        # To compute (X * Y / 2) % MOD safely:
        # One of X or Y must be even if X*Y is to be an integer after division by 2.
        # Here, m or m+1 is even.
        val1_m = m
        val2_m = m + 1
        
        if val1_m % 2 == 0:
            term1_mod = (val1_m // 2) % MOD
            term2_mod = val2_m % MOD
        else:
            term1_mod = val1_m % MOD
            term2_mod = (val2_m // 2) % MOD
        
        sum1 = (term1_mod * term2_mod) % MOD
            
    # count_rem is the number of remaining elements we need to pick.
    count_rem = n - m
    sum2 = 0
    
    if count_rem > 0:
        # These remaining numbers start from `target`.
        # They are: target, target + 1, ..., target + count_rem - 1.
        # Sum of an arithmetic series: N/2 * (first + last)
        # N = count_rem
        # first_val = target
        # last_val = target + count_rem - 1
        # Sum = count_rem / 2 * (target + target + count_rem - 1)
        # Sum = count_rem / 2 * (2*target + count_rem - 1)

        # Let val_a = count_rem
        # Let val_b = 2*target + count_rem - 1
        # One of val_a or val_b must be even.
        # If count_rem (val_a) is odd, then (2*target + count_rem - 1) is (even + odd - odd) = even.
        
        val_a_cr = count_rem
        val_b_cr = 2 * target + count_rem - 1 # This sum can be large

        if val_a_cr % 2 == 0:
            term_a_mod = (val_a_cr // 2) % MOD
            term_b_mod = val_b_cr % MOD 
        else: # val_b_cr must be even
            term_a_mod = val_a_cr % MOD
            term_b_mod = (val_b_cr // 2) % MOD
        
        sum2 = (term_a_mod * term_b_mod) % MOD
        
    total_sum = (sum1 + sum2) % MOD
    return total_sum