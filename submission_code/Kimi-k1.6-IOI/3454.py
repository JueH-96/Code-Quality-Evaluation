class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        res = 0
        prev = 0
        for d in map(lambda t, n: t - n, target, nums):
            if d > prev:
                res += d - prev
            prev = d
        return res