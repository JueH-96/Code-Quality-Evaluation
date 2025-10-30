class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # For each remainder when divided by k, track the minimum prefix sum
        min_prefix = {}
        max_sum = float('-inf')
        
        for j in range(n + 1):
            remainder = j % k
            
            # If we've seen this remainder before, we can form a valid subarray
            if remainder in min_prefix:
                subarray_sum = prefix[j] - min_prefix[remainder]
                max_sum = max(max_sum, subarray_sum)
            
            # Update the minimum prefix sum for this remainder
            if remainder not in min_prefix:
                min_prefix[remainder] = prefix[j]
            else:
                min_prefix[remainder] = min(min_prefix[remainder], prefix[j])
        
        return max_sum