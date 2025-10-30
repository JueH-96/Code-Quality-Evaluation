from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        for i in range(n - 2):  # i ranges from 0 to n-3
            for j in range(i + 1, n - 1):  # j ranges from i+1 to n-2
                current_cost = nums[0] + nums[i + 1] + nums[j + 1]
                if current_cost < min_cost:
                    min_cost = current_cost
        return min_cost