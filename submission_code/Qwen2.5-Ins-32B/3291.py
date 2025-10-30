from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            return bin(n).count('1')
        
        n = len(nums)
        sorted_nums = sorted(nums)
        bit_counts = [count_set_bits(num) for num in nums]
        
        i = 0
        while i < n:
            j = i
            while j < n and bit_counts[j] == bit_counts[i]:
                j += 1
            if not all(nums[k] <= nums[k+1] for k in range(i, j-1)):
                return False
            i = j
        
        return nums == sorted_nums