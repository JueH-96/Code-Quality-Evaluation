from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Step 1: Compute prefix_sum
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Step 2: Compute coefficients c_j for j from 1 to k
        c = []
        for j in range(1, k + 1):
            sign = 1 if (j % 2 == 1) else -1
            magnitude = k - j + 1
            c_j = sign * magnitude
            c.append(c_j)
        
        # Initialize DP arrays
        prev_dp = [0] * n  # dp[0][i] = 0 for all i
        curr_dp = [0] * n
        
        # Step 3: DP computation
        for j in range(1, k + 1):
            if j == 1:
                min_prefix = float('inf')
                for i in range(n):
                    min_prefix = min(min_prefix, prefix_sum[0:i+1])
                    curr_dp[i] = c[j - 1] * (prefix_sum[i + 1] - min_prefix)
            else:
                max_expr = float('-inf')
                for i in range(n):
                    if i == 0:
                        curr_dp[i] = float('-inf')
                    else:
                        curr_dp[i] = max_expr + c[j - 1] * prefix_sum[i + 1]
                    temp = prev_dp[i] - c[j - 1] * prefix_sum[i + 1]
                    max_expr = max(max_expr, temp)
            # Swap prev_dp and curr_dp for the next iteration
            prev_dp, curr_dp = curr_dp, prev_dp
        
        # The answer is the maximum value in prev_dp
        return max(prev_dp)