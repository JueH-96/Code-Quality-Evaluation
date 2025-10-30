from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        delta = [t - n for n, t in zip(nums, target)]
        res = 0
        prev = 0
        for d in delta:
            res += abs(d - prev)
            prev = d
        return res