from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums) - 2):
            # For subarray [nums[i], nums[i+1], nums[i+2]]:
            # We want: nums[i] + nums[i+2] == (nums[i+1] / 2)
            # Multiply both sides by 2 (works because sums are integers)
            if 2 * (nums[i] + nums[i+2]) == nums[i+1]:
                count += 1
        return count