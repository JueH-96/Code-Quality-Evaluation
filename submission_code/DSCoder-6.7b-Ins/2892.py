class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        base = [i for i in range(1, n) for _ in (1, 2)] + [n, n]
        nums.sort()
        return nums == base