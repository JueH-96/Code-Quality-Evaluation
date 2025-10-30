from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        for a in range(1, n-1):
            for b in range(a+1, n):
                current_sum = nums[a] + nums[b]
                if current_sum < min_sum:
                    min_sum = current_sum
        return nums[0] + min_sum