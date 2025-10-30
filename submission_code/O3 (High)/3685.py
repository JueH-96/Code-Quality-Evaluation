from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        """
        Counts the number of length-3 contiguous subarrays where
        (first element + third element) == (second element) / 2.

        Since all numbers are integers, the equality can be checked
        without division by multiplying both sides by 2:

            2 * (first + third) == middle
        """
        n = len(nums)
        count = 0
        
        for i in range(n - 2):
            first, middle, third = nums[i], nums[i + 1], nums[i + 2]
            if 2 * (first + third) == middle:
                count += 1
                
        return count