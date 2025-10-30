class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n < k:
            return 0
        # Initialize the prefix sum array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        # Initialize the maximum sum to negative infinity
        max_sum = -float('inf')
        # Iterate over all possible starting points
        for i in range(n - k + 1):
            current_sum = prefix[i + k] - prefix[i]
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum