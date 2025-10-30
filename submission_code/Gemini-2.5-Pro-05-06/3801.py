import sys
from functools import lru_cache

# It's good practice to increase recursion limit for competitive programming,
# though for L=9 (max number of digits for r < 10^9), default limit (1000) is fine.
# sys.setrecursionlimit(10**5) # Not strictly needed here for L_max=9.

class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        
        # sN_digits, L, TARGET_SUM will be set by _count_upto_N and accessed by _dp.
        # This avoids passing them recursively and including them in memoization keys for _dp.
        self.sN_digits = [] # Stores digits of N as integers
        self.L = 0
        self.TARGET_SUM = 0
        
        # Call count for r and l-1
        ans_r = self._count_upto_N(r)
        ans_l_minus_1 = self._count_upto_N(l - 1)
        
        return ans_r - ans_l_minus_1

    # Using lru_cache for memoization.
    # The cache is tied to the instance 'self' and the method '_dp'.
    # It's cleared appropriately in _count_upto_N when parameters like TARGET_SUM change.
    @lru_cache(maxsize=None)
    def _dp(self, idx: int, current_sum: int, current_prod_mod: int, tight: bool, is_leading_phase: bool, num_has_zero: bool) -> int:
        # idx: current digit index we are trying to place.
        # current_sum: sum of digits chosen so far for the actual number.
        # current_prod_mod: product of digits (modulo TARGET_SUM) chosen so far.
        # tight: boolean, True if we are restricted by the digits of N (sN_digits).
        # is_leading_phase: True if we are currently placing leading zeros or haven't started the number yet.
        # num_has_zero: True if a '0' has been placed as part of the actual number (not a leading zero).

        if idx == self.L:
            if is_leading_phase: # Indicates the number formed was effectively 0 (e.g. "0", "00"). Not a positive integer.
                return 0
            if current_sum != self.TARGET_SUM: # Sum of digits must match TARGET_SUM.
                return 0
            
            # Sum condition is met. Now check product condition.
            # TARGET_SUM is guaranteed to be >= 1 by the loop in _count_upto_N.
            if num_has_zero: 
                # If a zero was part of the number, product of digits is 0.
                # 0 % TARGET_SUM == 0 is true for TARGET_SUM >= 1.
                return 1
            
            # If no zero in number, product is non-zero. Check if current_prod_mod is 0.
            # This means (actual product) % TARGET_SUM == 0.
            if current_prod_mod == 0: # This check relies on TARGET_SUM > 0 for modulo arithmetic.
                return 1
            
            return 0 # Product is not divisible by sum.

        ans = 0
        # Determine the upper limit for the current digit.
        # If 'tight' is true, we can't choose a digit greater than sN_digits[idx].
        # Otherwise, we can choose up to 9.
        upper_bound = self.sN_digits[idx] if tight else 9

        for digit in range(upper_bound + 1):
            new_tight = tight and (digit == upper_bound) # New 'tight' for next state.
            
            if is_leading_phase:
                if digit == 0:
                    # Still in leading phase. Sum remains 0. Product of empty set of digits is 1.
                    ans += self._dp(idx + 1, 0, 1, new_tight, True, False)
                else: # digit > 0. This is the first non-zero digit. Number starts.
                    # Number is no longer in leading phase. num_has_zero is False.
                    # current_sum becomes `digit`. current_prod_mod becomes `digit % TARGET_SUM`.
                    # (TARGET_SUM is guaranteed >= 1).
                    ans += self._dp(idx + 1, digit, digit % self.TARGET_SUM, new_tight, False, False)
            
            else: # Not in leading phase (number has already started with a non-zero digit).
                # Optimization: if sum including current digit exceeds TARGET_SUM, prune this path.
                if current_sum + digit > self.TARGET_SUM:
                    continue # This path cannot lead to current_sum == TARGET_SUM.

                if num_has_zero: 
                    # Product is already 0 (due to a previous zero digit) and will remain 0.
                    # Pass current_prod_mod as 0 to keep state consistent.
                    ans += self._dp(idx + 1, current_sum + digit, 0, new_tight, False, True)
                else: # Product is not zero yet (all digits so far were non-zero).
                    if digit == 0:
                        # Current digit is 0. Product becomes 0. num_has_zero becomes True.
                        # Pass current_prod_mod as 0.
                        ans += self._dp(idx + 1, current_sum + digit, 0, new_tight, False, True)
                    else: # Current digit is also non-zero.
                        # Update product: (old_prod_mod * digit) % TARGET_SUM.
                        # (TARGET_SUM is guaranteed >= 1).
                        new_prod_mod = (current_prod_mod * digit) % self.TARGET_SUM
                        ans += self._dp(idx + 1, current_sum + digit, new_prod_mod, new_tight, False, False)
        
        return ans

    def _count_upto_N(self, N_val: int) -> int:
        if N_val == 0: # No positive beautiful numbers up to 0.
            return 0
            
        sN_str = str(N_val)
        self.sN_digits = [int(d) for d in sN_str] # Convert N to list of digits.
        self.L = len(self.sN_digits)
        
        total_count = 0
        
        # Iterate over all possible values for sum_of_digits (TARGET_SUM).
        # Max sum for a number with L digits is 9*L.
        # For N < 10^9, L_max = 9 (e.g., for 999,999,999). Max sum = 9*9 = 81.
        # Min sum is 1 (for positive integers).
        max_possible_sum_for_L_digits = 9 * self.L 

        for ts in range(1, max_possible_sum_for_L_digits + 1):
            self.TARGET_SUM = ts
            self._dp.cache_clear() # Crucial: clear cache for each new TARGET_SUM.
                                  # The behavior of _dp depends on self.TARGET_SUM.
            
            # Initial call to _dp for the number N_val:
            # idx=0, current_sum=0, current_prod_mod=1 (identity for product of empty set),
            # tight=True (restricted by N_val's digits),
            # is_leading_phase=True (haven't placed any non-zero digit yet),
            # num_has_zero=False (no zero encountered in the number yet).
            count_for_ts = self._dp(0, 0, 1, True, True, False)
            total_count += count_for_ts
            
        return total_count