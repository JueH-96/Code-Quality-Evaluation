class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find all positions of 1s
        ones_positions = []
        for i in range(len(nums)):
            if nums[i] == 1:
                ones_positions.append(i)
        
        # If no 1s, no valid splits
        if not ones_positions:
            return 0
        
        # If only one 1, only one way to split
        if len(ones_positions) == 1:
            return 1
        
        # Calculate the product of gaps between consecutive 1s
        result = 1
        for i in range(1, len(ones_positions)):
            gap = ones_positions[i] - ones_positions[i-1]
            result = (result * gap) % MOD
        
        return result