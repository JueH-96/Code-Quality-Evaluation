class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        if len(nums) != n + 1:
            return False
        expected = list(range(1, n)) + [n, n]
        return sorted(nums) == expected