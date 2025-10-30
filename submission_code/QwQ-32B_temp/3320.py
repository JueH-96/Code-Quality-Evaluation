from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        s = nums[0] + nums[1]
        count = 1
        current = 2
        while current <= n - 2:
            if nums[current] + nums[current + 1] == s:
                count += 1
                current += 2
            else:
                break
        return count