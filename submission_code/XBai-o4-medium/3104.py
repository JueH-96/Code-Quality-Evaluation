import bisect
from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        for k in range(n + 1):
            x = bisect.bisect_left(nums, k)
            if x == k and (k == n or (nums[k] if k < n else float('inf')) != k):
                ans += 1
        return ans