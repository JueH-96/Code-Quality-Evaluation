class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        # We'll use prefix sums and track the minimum prefix sum seen
        # for each remainder modulo k. If prefixSum[i] and prefixSum[j]
        # have the same remainder modulo k, then the subarray (j+1..i)
        # has length divisible by k, and its sum = prefixSum[i] - prefixSum[j].
        
        n = len(nums)
        
        # Compute prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
        
        # Initialize array to track the minimum prefix sum for each remainder
        import math
        min_prefix = [math.inf] * k
        min_prefix[0] = 0  # The prefix sum at index 0 (no elements) is 0
        
        max_sum = -math.inf
        
        # Iterate and compute the best possible subarray sum with length multiple of k
        for i in range(1, n + 1):
            r = i % k
            # Current prefix sum
            current_sum = prefix_sum[i]
            # Check the difference with the smallest prefix sum that has the same remainder
            candidate = current_sum - min_prefix[r]
            if candidate > max_sum:
                max_sum = candidate
            # Update the minimum prefix sum for this remainder
            if current_sum < min_prefix[r]:
                min_prefix[r] = current_sum
        
        return max_sum