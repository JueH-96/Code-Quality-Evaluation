import bisect
from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        for k in range(n + 1):
            idx = bisect.bisect_left(nums, k)
            # Check if k exists in nums
            if idx < n and nums[idx] == k:
                continue
            # The count of elements less than k is idx
            if idx == k:
                ans += 1
        return ans