class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        nums = [0] + nums + [0]
        prefix = [0] * (n + 2)
        suffix = [0] * (n + 2)
        for i in range(1, n + 1):
            prefix[i] = max(prefix[i - 1], nums[i - 1])
        for i in range(n, 0, -1):
            suffix[i] = max(suffix[i + 1], nums[i])
        return max(prefix[i] + suffix[i + 1] for i in range(n))