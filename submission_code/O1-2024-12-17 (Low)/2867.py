class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find all positions of 1's in the array
        ones = [i for i, val in enumerate(nums) if val == 1]
        
        # If there are no 1's, no valid splits
        if not ones:
            return 0
        
        # If there is only one '1', there's exactly 1 way (the whole array as one subarray)
        if len(ones) == 1:
            return 1
        
        ways = 1
        # Multiply the gaps between consecutive 1's
        for i in range(len(ones) - 1):
            ways = (ways * (ones[i+1] - ones[i])) % MOD
        
        return ways % MOD