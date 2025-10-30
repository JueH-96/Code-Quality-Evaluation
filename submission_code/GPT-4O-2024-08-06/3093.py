class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        def count_set_bits(n: int) -> int:
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count
        
        total_sum = 0
        for index, value in enumerate(nums):
            if count_set_bits(index) == k:
                total_sum += value
        
        return total_sum