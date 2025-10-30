class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find all indices where the element is 1
        ones = [i for i, val in enumerate(nums) if val == 1]
        
        # If no ones, no valid splits
        if not ones:
            return 0
        
        # If only one '1', there is exactly one way (the entire array as a single subarray)
        if len(ones) == 1:
            return 1
        
        # The number of ways is the product of (difference between consecutive ones)
        ways = 1
        for i in range(len(ones) - 1):
            ways = (ways * (ones[i+1] - ones[i])) % MOD
        
        return ways % MOD