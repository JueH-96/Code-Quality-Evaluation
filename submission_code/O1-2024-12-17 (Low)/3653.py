class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        # Compute prefix sums
        prefix_sum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # best[r] will store the minimum prefix_sum index encountered so far
        # for all indices i with i % k == r
        best = [float('inf')] * k
        best[0] = 0  # For subarrays starting from index 0
        
        max_sum = float('-inf')
        
        for i in range(1, len(nums) + 1):
            r = i % k
            # Calculate candidate subarray sum where subarray length is divisible by k
            max_sum = max(max_sum, prefix_sum[i] - best[r])
            # Update the minimum prefix sum encountered for remainder r
            best[r] = min(best[r], prefix_sum[i])
        
        return max_sum