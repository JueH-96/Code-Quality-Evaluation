class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find positions of all 1s
        ones_positions = [i for i, num in enumerate(nums) if num == 1]
        
        # If no 1s, can't form any good subarrays
        if not ones_positions:
            return 0
        
        # Calculate the product of distances between consecutive 1s
        result = 1
        for i in range(1, len(ones_positions)):
            distance = ones_positions[i] - ones_positions[i-1]
            result = (result * distance) % MOD
        
        return result