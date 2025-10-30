from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        # Iterate through all possible starting indices of subarrays of length 3
        for i in range(len(nums) - 2):
            a = nums[i]
            b = nums[i+1]
            c = nums[i+2]
            if 2 * (a + c) == b:
                count += 1
        return count