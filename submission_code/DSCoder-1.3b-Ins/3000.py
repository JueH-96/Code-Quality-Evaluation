class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        min_diff = float('inf')

        for i in range(n):
            prefix[i + 1] = min(prefix[i], nums[i])
            suffix[n - i] = min(suffix[n - i + 1], nums[n - i])

        for i in range(x, n - x + 1):
            min_diff = min(min_diff, abs(prefix[i - x] - suffix[i + x]))

        return min_diff