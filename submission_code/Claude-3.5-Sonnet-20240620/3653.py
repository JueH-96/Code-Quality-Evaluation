class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        
        # Calculate prefix sum array
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i-1] + nums[i-1]
        
        max_sum = float('-inf')
        min_prefix = {0: 0}  # Store minimum prefix sum for each remainder
        
        for i in range(k, n + 1):
            remainder = i % k
            if remainder in min_prefix:
                max_sum = max(max_sum, prefix_sum[i] - min_prefix[remainder])
            
            # Update minimum prefix sum for the current remainder
            current_prefix = prefix_sum[i - k + 1]
            if remainder not in min_prefix or current_prefix < min_prefix[remainder]:
                min_prefix[remainder] = current_prefix
        
        return max_sum