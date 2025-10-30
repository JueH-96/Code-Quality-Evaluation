from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        """
        Calculates the sum of power of all non-empty groups of heroes.
        The power of a group is defined as max(group)^2 * min(group).
        The result is returned modulo 10^9 + 7.
        """
        MOD = 10**9 + 7
        
        nums.sort()
        
        total_power = 0
        # s represents the sum of minimums of all non-empty subsequences of the prefix
        # of `nums` processed so far. Let the prefix be nums[0...i-1], s holds
        # sum(min(sub)) for all non-empty sub.
        s = 0
        
        for x in nums:
            # For each x = nums[i], we calculate its contribution as the maximum element.
            # A group with max x is {x} U G', where G' is a subset of {nums[0...i-1]}.
            # The contribution of x is x^2 * (sum of minimums of these groups).
            # The sum of minimums is (x + s).
            # So, contribution = x^2 * (x + s) = x^3 + x^2 * s.
            
            # We calculate this contribution modulo MOD.
            # Using intermediate modulo operations is good practice.
            x_sq = (x * x) % MOD
            term1 = (x_sq * x) % MOD      # x^3 mod MOD
            term2 = (x_sq * s) % MOD      # x^2 * s mod MOD
            contribution = (term1 + term2) % MOD
            
            total_power = (total_power + contribution) % MOD
            
            # Update s for the next iteration using the recurrence:
            # s_new = 2 * s_old + x
            s = (s * 2 + x) % MOD
            
        return total_power