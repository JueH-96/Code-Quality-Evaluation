from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Record the indices of all 1's in the array
        positions = [i for i, v in enumerate(nums) if v == 1]
        
        # If there are no 1's, we can't split into good subarrays
        if not positions:
            return 0
        
        # We will multiply the gaps between consecutive 1's
        res = 1
        for i in range(len(positions) - 1):
            # The number of ways to choose the split between positions[i] and positions[i+1]
            gap = positions[i+1] - positions[i]
            res = (res * gap) % MOD
        
        return res