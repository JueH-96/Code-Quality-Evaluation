from typing import List
import sys

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        # Compute coefficients c[1..k]
        c = [0] * (k + 1)
        for j in range(1, k + 1):
            sign = 1 if j % 2 == 1 else -1
            coeff = k - j + 1
            c[j] = sign * coeff
        
        # Initialize previous DP array (j-1 subproblems)
        prev_dp = [0] * (n + 1)
        
        for j in range(1, k + 1):
            curr_dp = [-float('inf')] * (n + 1)
            curr_max = -float('inf')
            for i in range(1, n + 1):
                # Calculate temp = prev_dp[i-1] - c[j] * prefix[i-1]
                temp = prev_dp[i-1] - c[j] * prefix[i-1]
                if temp > curr_max:
                    curr_max = temp
                option2 = curr_max + c[j] * prefix[i]
                # curr_dp[i] is the maximum of previous curr_dp[i-1] and option2
                if i == 1:
                    curr_dp[i] = option2
                else:
                    curr_dp[i] = max(curr_dp[i-1], option2)
            # Update prev_dp to curr_dp for the next iteration
            prev_dp = curr_dp
        
        return prev_dp[n]