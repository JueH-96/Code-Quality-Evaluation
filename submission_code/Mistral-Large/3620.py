from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        if k == 0:
            return len(set(nums))

        nums.sort()
        n = len(nums)
        i = 0

        while i < n and (i == 0 or nums[i] - nums[i - 1] <= 2 * k):
            i += 1

        return n - i + 1