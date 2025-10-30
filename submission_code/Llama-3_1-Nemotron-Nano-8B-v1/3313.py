from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        coeff = [0] * (k + 1)
        for j in range(1, k + 1):
            sign = (-1) ** (j + 1)
            coeff[j] = sign * (k - j + 1)
        
        prev_dp = [-float('inf')] * (n + 1)
        prev_dp[0] = 0  # Base case: 0 subarrays selected, strength 0
        
        for j in range(1, k + 1):
            curr_dp = [-float('inf')] * (n + 1)
            if j == 1:
                curr_dp[0] = 0
                min_prefix = prefix_sum[0]
                for i in range(1, n + 1):
                    # Calculate the maximum possible sum for a single subarray ending at i
                    curr_dp[i] = max(curr_dp[i - 1], (prefix_sum[i] - min_prefix) * coeff[j])
                    # Update the minimum prefix sum encountered so far
                    min_prefix = min(min_prefix, prefix_sum[i])
            else:
                current_max = -float('inf')
                for i in range(1, n + 1):
                    # Update the current_max with the best value from previous positions
                    current_max = max(current_max, prev_dp[i - 1] - prefix_sum[i - 1] * coeff[j])
                    # Option 1: Do not include the current element in the j-th subarray
                    option1 = curr_dp[i - 1]
                    # Option 2: Include the current element in the j-th subarray
                    option2 = prefix_sum[i] * coeff[j] + current_max
                    curr_dp[i] = max(option1, option2)
            prev_dp = curr_dp.copy()
        
        return max(prev_dp)