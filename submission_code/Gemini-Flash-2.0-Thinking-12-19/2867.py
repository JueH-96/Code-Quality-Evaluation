class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        indices_of_ones = [i for i, num in enumerate(nums) if num == 1]
        num_ones = len(indices_of_ones)
        if num_ones == 0:
            return 0
        if num_ones == 1:
            return 1
        
        MOD = 10**9 + 7
        result = 1
        for i in range(num_ones - 1):
            diff = indices_of_ones[i+1] - indices_of_ones[i]
            result = (result * diff) % MOD
            
        return result