from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n - 2):
            current_sum = nums[i] + nums[i + 2]
            if 2 * current_sum == nums[i + 1]:
                count += 1
        return count