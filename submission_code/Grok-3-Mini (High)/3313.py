from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Compute prefix sum
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num)
        
        # Compute coefficients
        coef = [0] * k
        for idx in range(k):
            coef[idx] = ((-1) ** idx) * (k - idx)
        
        # Define a large negative number for -infinity
        NEG_INF = -10000000000000000000000  # -1e22, sufficiently small to handle minimum possible strength
        
        # Create dp table: dp[idx][i] for idx=0 to k-1, i=0 to n-1
        dp = [[0 for _ in range(n)] for _ in range(k)]
        
        # Handle the first subarray (idx=0)
        # Compute min prefix sum values
        min_prefix_sum_val = [prefix_sum[0]]
        for t in range(1, n + 1):
            min_prefix_sum_val.append(min(min_prefix_sum_val[-1], prefix_sum[t]))
        
        for i in range(n):
            dp[0][i] = coef[0] * (prefix_sum[i + 1] - min_prefix_sum_val[i])
        
        # Now for each additional subarray from idx=1 to k-1
        for idx in range(1, k):
            # Compute cumulative max for p < s
            cum_max_list = [NEG_INF] * (n + 1)
            cum_max_list[1] = dp[idx - 1][0]
            for s in range(2, n + 1):
                cum_max_list[s] = max(cum_max_list[s - 1], dp[idx - 1][s - 1])
            
            # Compute g[s] for s=0 to n-1
            g = [0] * n
            for s in range(n):
                g[s] = -coef[idx] * prefix_sum[s] + cum_max_list[s]
            
            # Compute prefix max of g
            prefix_max_g = [g[0]]
            for t in range(1, n):
                prefix_max_g.append(max(prefix_max_g[-1], g[t]))
            
            # Now compute dp[idx][i] for each i
            for i in range(n):
                dp[idx][i] = coef[idx] * prefix_sum[i + 1] + prefix_max_g[i]
        
        # The answer is the maximum value in the last dp row
        return max(dp[k - 1])