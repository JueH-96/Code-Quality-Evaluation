from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_total = float('inf')
        for i in range(n - 2):  # i can be from 0 to n-3 inclusive
            for j in range(i + 1, n - 1):  # j from i+1 to n-2 inclusive
                current_sum = nums[i + 1] + nums[j + 1]
                total = nums[0] + current_sum
                if total < min_total:
                    min_total = total
        return min_total