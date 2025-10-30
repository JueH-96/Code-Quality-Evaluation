class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')
        
        # Prefix sum array to store cumulative sums
        prefix_sum = [0] * (n + 1)
        
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Use a dictionary to store the minimum prefix sum for each remainder when divided by k
        min_prefix_sum = {0: 0}
        
        for i in range(1, n + 1):
            remainder = i % k
            
            # If we have seen this remainder before, we can calculate the sum of the subarray
            if remainder in min_prefix_sum:
                current_sum = prefix_sum[i] - min_prefix_sum[remainder]
                max_sum = max(max_sum, current_sum)
            
            # Update the minimum prefix sum for this remainder
            if remainder not in min_prefix_sum:
                min_prefix_sum[remainder] = prefix_sum[i]
        
        return max_sum