class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        max_sum = float('-inf')
        
        # For each remainder, keep track of the minimum prefix sum seen so far
        min_prefix = [float('inf')] * k
        
        for j in range(n + 1):
            remainder = j % k
            # If we've seen this remainder before, calculate potential max sum
            if min_prefix[remainder] != float('inf'):
                max_sum = max(max_sum, prefix[j] - min_prefix[remainder])
            # Update minimum prefix sum for this remainder
            min_prefix[remainder] = min(min_prefix[remainder], prefix[j])
        
        return max_sum