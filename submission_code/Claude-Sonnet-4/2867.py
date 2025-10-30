class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find all positions of 1s
        ones_positions = []
        for i in range(len(nums)):
            if nums[i] == 1:
                ones_positions.append(i)
        
        # If no 1s, impossible to create good subarrays
        if not ones_positions:
            return 0
        
        # If only one 1, there's exactly one way (the entire array)
        if len(ones_positions) == 1:
            return 1
        
        # For each pair of consecutive 1s, count the ways to split between them
        result = 1
        for i in range(len(ones_positions) - 1):
            # Distance between consecutive 1s
            distance = ones_positions[i + 1] - ones_positions[i]
            # Number of ways to place the split = distance (we can split after any position between them)
            result = (result * distance) % MOD
        
        return result