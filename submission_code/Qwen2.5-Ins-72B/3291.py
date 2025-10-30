from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            return bin(n).count('1')
        
        def can_sort(nums):
            n = len(nums)
            for i in range(n - 1):
                if nums[i] > nums[i + 1]:
                    return False
            return True
        
        def swap_adjacent(nums):
            n = len(nums)
            for i in range(n - 1):
                if count_set_bits(nums[i]) == count_set_bits(nums[i + 1]):
                    nums[i], nums[i + 1] = nums[i + 1], nums[i]
        
        sorted_nums = sorted(nums)
        while nums != sorted_nums:
            swapped = False
            for i in range(len(nums) - 1):
                if nums[i] > nums[i + 1] and count_set_bits(nums[i]) == count_set_bits(nums[i + 1]):
                    nums[i], nums[i + 1] = nums[i + 1], nums[i]
                    swapped = True
            if not swapped:
                break
        
        return can_sort(nums)