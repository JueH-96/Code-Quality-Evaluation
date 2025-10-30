from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        INF = float('-inf')
        # Initialize DP table with dimensions (k+1) x (n+1)
        dp = [[INF] * (n + 1) for _ in range(k + 1)]
        
        # Base case: 0 subarrays, sum is 0 for any i
        for i in range(n + 1):
            dp[0][i] = 0
        
        for j in range(1, k + 1):
            # Compute prev_val array for the previous j-1 subarrays
            prev_val = [INF] * (n + 1)
            for s in range(n + 1):
                prev_val[s] = dp[j-1][s] - prefix[s]
            
            # Compute max_prev array which holds the maximum of prev_val up to each index
            max_prev = [INF] * (n + 1)
            max_prev[0] = prev_val[0]
            for t in range(1, n + 1):
                max_prev[t] = max(max_prev[t-1], prev_val[t])
            
            # Compute dp[j][i] for each i from 0 to n
            for i in range(n + 1):
                if i < j * m:
                    current = INF
                else:
                    # The candidate value using the current subarray ending at i
                    candidate = max_prev[i - m] + prefix[i]
                    if i == 0:
                        current = candidate
                    else:
                        current = max(dp[j][i-1], candidate)
                dp[j][i] = current
        
        return dp[k][n]