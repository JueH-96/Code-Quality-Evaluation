class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i] * (1 if i % 2 == 0 else -1)
            suffix[n - i] = suffix[n - i + 1] + nums[n - i - 1] * (1 if (n - i - 1) % 2 == 0 else -1)
        total = min(prefix[i] + suffix[i + 1] for i in range(n + 1))
        return max(total, max(prefix[i] + suffix[i] for i in range(n)))