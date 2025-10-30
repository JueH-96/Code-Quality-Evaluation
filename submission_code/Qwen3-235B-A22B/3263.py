from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_total = float('inf')
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                current_sum = nums[0] + nums[i + 1] + nums[j + 1]
                if current_sum < min_total:
                    min_total = current_sum
        return min_total