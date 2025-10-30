from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        """
        Count the number of subarrays of length 3 such that
        the sum of the first and third elements equals
        exactly half of the second element.
        """
        n = len(nums)
        count = 0
        
        # Loop through all subarrays of length 3
        for i in range(n - 2):
            first, mid, third = nums[i], nums[i + 1], nums[i + 2]
            # Check if (first + third) == mid / 2,
            # i.e., 2*(first + third) == mid
            if 2 * (first + third) == mid:
                count += 1
        
        return count