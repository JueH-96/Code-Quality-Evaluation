class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        def count_set_bits(n: int) -> int:
            return bin(n).count('1')
        
        total_sum = 0
        for i in range(len(nums)):
            if count_set_bits(i) == k:
                total_sum += nums[i]
        
        return total_sum