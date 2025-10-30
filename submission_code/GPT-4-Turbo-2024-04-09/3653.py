class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')
        
        # Calculate prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # We use a dictionary to store the minimum prefix sum for each modulo k result
        min_prefix_sum = {}
        
        # Iterate over the prefix sums to find the maximum subarray sum with length divisible by k
        for i in range(n + 1):
            mod = i % k
            if mod in min_prefix_sum:
                # Calculate subarray sum with length divisible by k
                if (i - min_prefix_sum[mod]) % k == 0:
                    max_sum = max(max_sum, prefix_sum[i] - prefix_sum[min_prefix_sum[mod]])
            else:
                # Store the first occurrence of this mod result
                min_prefix_sum[mod] = i
        
        return max_sum