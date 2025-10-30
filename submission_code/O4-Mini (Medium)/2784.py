from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        
        pref = 0   # prefix sum of mins over all non-empty subsets of processed elements
        ans = 0
        
        for x in nums:
            # For each x as the maximum of a group, the sum of min over all non-empty
            # subsets of earlier elements is `pref`, and the empty subset contributes min = x.
            # So total min-sum for groups ending in x is (pref + x).
            # Each group's power is x^2 * min, so contribution is x^2 * (pref + x).
            ans = (ans + x * x % MOD * (pref + x) % MOD) % MOD
            
            # Update pref for next rounds:
            # pref[i] = 2*pref[i-1] + x  (because each existing subset can either include x
            # or not, and subsets including x contribute min=x for the empty base case
            # and inherit the old min for non-empty old subsets).
            pref = (2 * pref + x) % MOD
        
        return ans