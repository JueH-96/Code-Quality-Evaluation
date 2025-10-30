class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_counts = [0] * (n + 1)
        prefix_sum = 0
        for i in range(n):
            prefix_counts[i + 1] = prefix_counts[i] + i + 1 - (prefix_counts[i] - (nums[i] <= 100 and nums[i] >= 1))
            prefix_sum += prefix_counts[i + 1]
        return prefix_sum