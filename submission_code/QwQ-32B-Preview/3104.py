from typing import List
import bisect

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        result = 0
        for k in range(n + 1):
            count = bisect.bisect_left(nums, k)
            if count == k:
                # Check if there are no students with nums[i] = k
                if count == n or nums[count] != k:
                    result += 1
        return result