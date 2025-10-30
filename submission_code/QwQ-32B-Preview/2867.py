class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ones_positions = [i for i, num in enumerate(nums) if num == 1]
        
        if not ones_positions:
            return 0
        
        if len(ones_positions) == 1:
            return 1
        
        total_ways = 1
        for i in range(1, len(ones_positions)):
            z = ones_positions[i] - ones_positions[i - 1] - 1
            total_ways = (total_ways * (z + 1)) % MOD
        
        return total_ways