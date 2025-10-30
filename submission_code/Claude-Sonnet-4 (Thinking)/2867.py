class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find positions of all 1s
        ones_positions = []
        for i, num in enumerate(nums):
            if num == 1:
                ones_positions.append(i)
        
        # If no 1s, impossible to split into good subarrays
        if not ones_positions:
            return 0
        
        # If only one 1, only one way (whole array)
        if len(ones_positions) == 1:
            return 1
        
        # Calculate number of ways
        ways = 1
        for i in range(len(ones_positions) - 1):
            # Between consecutive 1s, we have multiple positions to place the cut
            choices = ones_positions[i+1] - ones_positions[i]
            ways = (ways * choices) % MOD
        
        return ways