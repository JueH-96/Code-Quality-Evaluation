from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute prefix sum of alternating nums
        prefix_sum = [0] * (n + 1)
        cum = 0
        for i in range(n):
            sign_alt = 1 if i % 2 == 0 else -1  # (-1)^i
            alt = nums[i] * sign_alt
            cum += alt
            prefix_sum[i + 1] = cum  # prefix_sum[k] = sum_{j=0}^{k-1} nums[j] * (-1)^j
        
        # Initialize dp and suffix max arrays
        dp = [0] * (n + 1)  # dp[n] = 0
        max_A_suffix = [0] * (n + 1)
        max_B_suffix = [0] * (n + 1)
        
        # Set values for k = n
        A_n = dp[n] + prefix_sum[n]
        B_n = dp[n] - prefix_sum[n]
        max_A_suffix[n] = A_n
        max_B_suffix[n] = B_n
        
        # Compute dp from n-1 to 0
        for i in range(n - 1, -1, -1):  # i from n-1 to 0 inclusive
            c_i = 1 if i % 2 == 0 else -1  # (-1)^i
            if c_i == 1:
                max_val = max_A_suffix[i + 1]
            else:
                max_val = max_B_suffix[i + 1]
            dp[i] = -c_i * prefix_sum[i] + max_val
            
            # Compute A and B for current i
            A_i = dp[i] + prefix_sum[i]
            B_i = dp[i] - prefix_sum[i]
            
            # Update max suffix values for current i
            max_A_suffix[i] = max(A_i, max_A_suffix[i + 1])
            max_B_suffix[i] = max(B_i, max_B_suffix[i + 1])
        
        return dp[0]