from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_part = float('inf')
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                current = nums[i + 1] + nums[j + 1]
                if current < min_part:
                    min_part = current
        return nums[0] + min_part