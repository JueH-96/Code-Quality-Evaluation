from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        delta = [0] * (n + 1)
        for l, r in queries:
            delta[l] += 1
            delta[r + 1] -= 1
        current = 0
        for i in range(n):
            current += delta[i]
            if current < nums[i]:
                return False
        return True