from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        prefix = [0]
        for num in nums_sorted:
            prefix.append(prefix[-1] + num)
        for i in range(n, 2, -1):
            if prefix[i-1] > nums_sorted[i-1]:
                return prefix[i]
        return -1