import sys

# Increase recursion limit to handle potential deep recursion for s.length up to 800
sys.setrecursionlimit(2000)

class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        N_len = len(s)
        s_int_arr = [int(c) for c in s] # Convert binary string s to a list of integers [1, 1, 1] for "111"

        # Precompute min_steps_to_1[i]: minimum operations to reduce integer i to 1.
        # The maximum relevant value for `i` here is N_len, as it's the maximum possible popcount of a number up to N.
        MAX_RELEVANT_POPCOUNT = N_len 
        
        min_steps_to_1 = [float('inf')] * (MAX_RELEVANT_POPCOUNT + 1)
        min_steps_to_1[1] = 0 # 1 requires 0 steps to become 1

        # Calculate min_steps_to_1 for values from 2 up to MAX_RELEVANT_POPCOUNT.
        # This loop correctly computes values in increasing order, as popcount(i) < i for i > 1.
        for i in range(2, MAX_RELEVANT_POPCOUNT + 1):
            # bin(i).count('1') calculates popcount of i
            min_steps_to_1[i] = 1 + min_steps_to_1[bin(i).count('1')]
        
        # Determine `valid_popcounts_of_x`: a set of popcount values `p`
        # such that a number `x` with `popcount(x) = p` is k-reducible.
        # An integer `x` is k-reducible if `steps(x) <= k`.
        # If `x=1`, `steps(x)=0`, so it's k-reducible for any `k >= 0`.
        # If `x > 1`, `steps(x) = 1 + steps(popcount(x))`.
        # So we need `1 + min_steps_to_1[popcount(x)] <= k`, which simplifies to `min_steps_to_1[popcount(x)] <= k - 1`.
        # The value `popcount(x)` can range from 1 to MAX_RELEVANT_POPCOUNT.
        
        valid_popcounts_of_x = set()
        for p_val in range(1, MAX_RELEVANT_POPCOUNT + 1):
            # Check if `p_val` makes `x` k-reducible
            # If p_val is 1, min_steps_to_1[1] is 0. Condition is 0 <= k-1, i.e., k >= 1.
            # Since problem constraint is k >= 1, 1 is always a valid popcount if it comes from an x>1.
            # If x=1, its popcount is 1, and it's 0-reducible. So if k >= 0, it's valid.
            # Our `valid_popcounts_of_x` will contain 1 if `k-1 >= 0`, which is true for `k >= 1`.
            if min_steps_to_1[p_val] <= k - 1:
                valid_popcounts_of_x.add(p_val)

        # Memoization dictionary for the digit DP
        memo = {}

        # dp(idx, current_popcount, is_less, is_leading_zero)
        # idx: Current bit position (0 to N_len-1)
        # current_popcount: Number of set bits in the number formed so far
        # is_less: True if the number formed is already strictly less than prefix of `s`
        # is_leading_zero: True if we are still placing leading zeros (number is effectively 0)
        def dp(idx, current_popcount, is_less, is_leading_zero):
            # Optimization: If current_popcount exceeds MAX_RELEVANT_POPCOUNT,
            # it cannot be in `valid_popcounts_of_x`.
            if current_popcount > MAX_RELEVANT_POPCOUNT:
                return 0
            
            # Base case: All bits have been placed.
            if idx == N_len:
                # If `is_leading_zero` is True, it means the number formed is 0 (e.g., "000").
                # We only count positive integers.
                if is_leading_zero:
                    return 0
                
                # Check if the popcount of the formed number is in our set of valid popcounts.
                return 1 if current_popcount in valid_popcounts_of_x else 0

            state = (idx, current_popcount, is_less, is_leading_zero)
            if state in memo:
                return memo[state]

            ans = 0
            # Determine the upper limit for the current digit.
            # If `is_less` is True, we can place '0' or '1'.
            # Otherwise, we are constrained by the digit in `s_int_arr[idx]`.
            upper_limit = s_int_arr[idx] if not is_less else 1

            for digit in range(upper_limit + 1):
                if is_leading_zero and digit == 0:
                    # If we are currently placing leading zeros and choose to place a '0',
                    # the popcount doesn't change, and we remain in leading zero mode.
                    # `is_less` becomes True because `s[0]` is guaranteed to be '1' (no leading zeros in `s`),
                    # so '0' is always less than `s[0]`.
                    ans = (ans + dp(idx + 1, current_popcount, True, True)) % MOD
                else:
                    # If we place a '1' (digit=1), or if we already placed a non-zero digit
                    # (`is_leading_zero` is False), then `current_popcount` increases by `digit`.
                    # `new_is_less` is true if `is_less` was already true, or if we placed a `digit`
                    # smaller than `s_int_arr[idx]`.
                    new_is_less = is_less or (digit < upper_limit)
                    ans = (ans + dp(idx + 1, current_popcount + digit, new_is_less, False)) % MOD

            memo[state] = ans
            return ans

        # The DP function `dp(0, 0, False, True)` counts all positive integers `x`
        # such that `0 < x <= n` and `x` is k-reducible.
        total_count_up_to_n = dp(0, 0, False, True)

        # The problem asks for positive integers LESS THAN n.
        # So, we need to subtract `1` if `n` itself is k-reducible.
        
        # Calculate popcount of n.
        popcount_n = s.count('1')
        
        is_n_reducible = False
        # Special case for n=1: It has 0 steps, so it's always k-reducible for k >= 1.
        if N_len == 1 and s_int_arr[0] == 1: # This means n is "1"
            is_n_reducible = True
        else:
            # For n > 1, n is k-reducible if its popcount is in `valid_popcounts_of_x`.
            # This is because `valid_popcounts_of_x` contains `p` such that `1 + min_steps_to_1[p] <= k`.
            if popcount_n in valid_popcounts_of_x:
                is_n_reducible = True

        # Subtract 1 (modulo MOD) if `n` itself is k-reducible, as we only count `x < n`.
        if is_n_reducible:
            total_count_up_to_n = (total_count_up_to_n - 1 + MOD) % MOD

        return total_count_up_to_n