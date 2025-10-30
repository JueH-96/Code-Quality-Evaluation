class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        base_n = list(range(1, n)) + [n, n]
        return sorted(nums) == sorted(base_n)