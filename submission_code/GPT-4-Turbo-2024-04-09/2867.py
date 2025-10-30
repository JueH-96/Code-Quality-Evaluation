class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ones_indices = [i for i, num in enumerate(nums) if num == 1]
        if len(ones_indices) < 2:
            return 0
        
        count = 0
        for i in range(1, len(ones_indices)):
            left_split_options = ones_indices[i] - ones_indices[i-1]
            count += left_split_options
            count %= MOD
        
        return count