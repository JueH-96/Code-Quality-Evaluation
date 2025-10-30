class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        if len(nums) != n + 1:
            return False
        base_n = list(range(1, n)) + [n, n]
        return sorted(nums) == sorted(base_n)