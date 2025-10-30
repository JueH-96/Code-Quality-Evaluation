class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(x):
            return bin(x).count('1')
        
        original_bits = [count_set_bits(num) for num in nums]
        sorted_nums = sorted(nums)
        sorted_bits = [count_set_bits(num) for num in sorted_nums]
        
        return original_bits == sorted_bits