from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        # We need to check each subarray of length 3.
        # The condition is 2*(first + third) equals the middle element.
        for i in range(len(nums) - 2):
            if 2 * (nums[i] + nums[i+2]) == nums[i+1]:
                count += 1
        return count