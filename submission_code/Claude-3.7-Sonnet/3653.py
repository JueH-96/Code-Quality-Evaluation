class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        prefix_sum = 0
        max_sum = float('-inf')
        min_prefix_sums = {}  # key: remainder, value: minimum prefix sum
        
        # Initialize for subarrays starting from the beginning
        min_prefix_sums[0] = 0
        
        for i in range(n):
            prefix_sum += nums[i]
            curr_remainder = (i + 1) % k
            
            # Check if there's a previous index with the same remainder
            # If yes, we found a subarray with length divisible by k
            if curr_remainder in min_prefix_sums:
                max_sum = max(max_sum, prefix_sum - min_prefix_sums[curr_remainder])
            
            # Update the minimum prefix sum for the current remainder
            min_prefix_sums[curr_remainder] = min(min_prefix_sums.get(curr_remainder, float('inf')), prefix_sum)
        
        return max_sum