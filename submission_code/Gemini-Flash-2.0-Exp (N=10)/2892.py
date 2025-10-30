from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        if len(nums) != n + 1:
            return False
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        if counts.get(n, 0) != 2:
            return False
        for i in range(1, n):
            if counts.get(i, 0) != 1:
                return False
        return True