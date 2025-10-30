from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        for i in range(n - 1, 1, -1):
            sum_before = prefix_sum[i]
            if sum_before > nums[i]:
                return sum_before + nums[i]
        return -1