from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        n = len(nums)
        averages = []
        for i in range(n // 2):
            # Pair the ith smallest with the ith largest
            avg = (nums[i] + nums[n - 1 - i]) / 2
            averages.append(avg)
        return min(averages)