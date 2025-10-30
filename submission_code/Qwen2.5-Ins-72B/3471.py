from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        averages = []
        for i in range(len(nums) // 2):
            minElement = nums[i]
            maxElement = nums[-(i + 1)]
            averages.append((minElement + maxElement) / 2)
        return min(averages)