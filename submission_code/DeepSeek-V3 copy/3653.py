class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n < k:
            return 0
        # Compute the prefix sum array
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
        max_sum = -float('inf')
        # Iterate over all possible starting points
        for i in range(n - k + 1):
            current_sum = prefix_sum[i + k] - prefix_sum[i]
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum