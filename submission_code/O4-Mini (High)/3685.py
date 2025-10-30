from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        # Iterate over all subarrays of length 3
        for i in range(len(nums) - 2):
            # Check if first + third equals half of the middle element
            # Multiply both sides by 2 to avoid floats:
            # 2 * (nums[i] + nums[i+2]) == nums[i+1]
            if 2 * (nums[i] + nums[i+2]) == nums[i+1]:
                count += 1
        return count