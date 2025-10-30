from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ones = [i for i, val in enumerate(nums) if val == 1]
        
        # If there are no ones, we cannot form any good subarray.
        if not ones:
            return 0
        
        # If there is exactly one 1, the entire array is the only valid good subarray.
        if len(ones) == 1:
            return 1
        
        ways = 1
        # For each adjacent pair of ones, compute how many ways to split between them.
        for i in range(1, len(ones)):
            # The number of zeros between the two ones is ones[i] - ones[i-1] - 1.
            # The number of ways to split between these two ones is (gap + 1)
            gap = ones[i] - ones[i - 1] - 1
            ways = (ways * (gap + 1)) % MOD
        
        return ways