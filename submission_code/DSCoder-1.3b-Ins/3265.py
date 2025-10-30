class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        max_sum = 0
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
            if i >= k - 1:
                max_sum = max(max_sum, prefix_sum[i + 1] - prefix_sum[i - k + 1])
        return max_sum