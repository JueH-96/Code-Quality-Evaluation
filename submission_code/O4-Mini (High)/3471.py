from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        n = len(nums)
        min_avg = float('inf')
        for i in range(n // 2):
            avg = (nums[i] + nums[n - 1 - i]) / 2.0
            if avg < min_avg:
                min_avg = avg
        return min_avg