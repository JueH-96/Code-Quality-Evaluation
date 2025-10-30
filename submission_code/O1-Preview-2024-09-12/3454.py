class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        res = 0
        prev = 0
        for curr in [t - n for n, t in zip(nums, target)]:
            res += abs(curr - prev)
            prev = curr
        return res