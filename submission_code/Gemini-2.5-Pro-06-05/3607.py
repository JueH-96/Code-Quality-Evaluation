import math
from typing import List

class Solution:
    """
    Solves the problem of finding the minimum operations to make an array non-decreasing.
    """
    
    # Precompute least prime factors using a linear sieve.
    # This block runs only once when the class is loaded, making it efficient
    # for multiple test cases.
    MAX_VAL = 10**6 + 1
    lpf = [0] * MAX_VAL
    primes = []
    if lpf[2] == 0:  # A check to ensure this runs only once
        lpf[1] = 1
        for i in range(2, MAX_VAL):
            if lpf[i] == 0:
                lpf[i] = i
                primes.append(i)
            for p in primes:
                if p > lpf[i] or i * p >= MAX_VAL:
                    break
                lpf[i * p] = p

    def minOperations(self, nums: List[int]) -> int:
        """
        Calculates the minimum operations to make the array non-decreasing.

        The operation is to divide an element by its greatest proper divisor,
        which is equivalent to replacing the element with its least prime factor.
        """
        n = len(nums)
        if n <= 1:
            return 0
            
        inf = n + 1
        
        # dp_prev_0: min ops for prefix ending at i-1, with nums[i-1] unchanged.
        # dp_prev_1: min ops for prefix ending at i-1, with nums[i-1] changed to its LPF.
        
        dp_prev_0 = 0
        n0 = nums[0]
        l0 = self.lpf[n0]
        # An operation on nums[0] is possible only if it's composite.
        dp_prev_1 = 1 if l0 < n0 else inf

        for i in range(1, n):
            ni = nums[i]
            li = self.lpf[ni]
            
            n_prev = nums[i-1]
            l_prev = self.lpf[n_prev]
            
            # --- Calculate dp_curr_0 (don't operate on nums[i]) ---
            dp_curr_0 = inf
            # Valid if previous value <= ni
            if n_prev <= ni:
                dp_curr_0 = min(dp_curr_0, dp_prev_0)
            if l_prev < n_prev and l_prev <= ni:
                dp_curr_0 = min(dp_curr_0, dp_prev_1)

            # --- Calculate dp_curr_1 (operate on nums[i]) ---
            dp_curr_1 = inf
            # Possible only if nums[i] is composite (li < ni)
            if li < ni:
                cost = 1
                prev_ops = inf
                # Valid if previous value <= li
                if n_prev <= li:
                    prev_ops = min(prev_ops, dp_prev_0)
                if l_prev < n_prev and l_prev <= li:
                    prev_ops = min(prev_ops, dp_prev_1)
                
                if prev_ops != inf:
                    dp_curr_1 = cost + prev_ops
            
            # Update dp states for the next iteration
            dp_prev_0 = dp_curr_0
            dp_prev_1 = dp_curr_1
            
        result = min(dp_prev_0, dp_prev_1)
        
        return result if result != inf else -1