from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(x):
            return bin(x).count('1')
        
        n = len(nums)
        sorted_nums = sorted(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] > nums[j] and count_set_bits(nums[i]) == count_set_bits(nums[j]):
                    nums[i], nums[j] = nums[j], nums[i]
        
        return nums == sorted_nums