from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i-1] + nums[i-1]
        # Check from the largest possible m down to 3
        for m in range(n, 2, -1):
            if prefix[m-1] > nums[m-1]:
                return prefix[m]
        return -1