class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ones_positions = [i for i, num in enumerate(nums) if num == 1]
        k = len(ones_positions)
        if k == 0:
            return 0
        if k == 1:
            return 1
        
        total_ways = 1
        for i in range(k - 1):
            gap = ones_positions[i+1] - ones_positions[i]
            total_ways = (total_ways * gap) % MOD
            
        return total_ways