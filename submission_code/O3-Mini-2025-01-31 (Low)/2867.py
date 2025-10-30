from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # collect indexes of ones
        ones = [i for i, val in enumerate(nums) if val == 1]
        
        # If there are no ones, we can't form any good subarray (because each must contain exactly one 1)
        if not ones:
            return 0
        
        # There is only one way to split when there is exactly one '1', because the entire array is a good subarray.
        if len(ones) == 1:
            return 1
        
        ways = 1
        # For every gap between consecutive ones, the number of ways to split is the number of zeros between them plus one.
        for i in range(1, len(ones)):
            gap = ones[i] - ones[i-1] - 1
            ways = (ways * (gap + 1)) % MOD
            
        return ways