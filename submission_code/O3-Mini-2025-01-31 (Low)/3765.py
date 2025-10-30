from typing import List
import math

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        # Precompute prefix sums: prefix sum starting with 0 index.
        preN = [0] * (n + 1)
        preC = [0] * (n + 1)
        for i in range(n):
            preN[i+1] = preN[i] + nums[i]
            preC[i+1] = preC[i] + cost[i]
        
        # dp[i][m] represents the minimum cost to partition first i elements into exactly m segments.
        # We only need dp for i up to n and m up to i.
        dp = [[math.inf]*(n+1) for _ in range(n+1)]
        dp[0][0] = 0
        
        # For each ending position i (1-indexed notion for prefix sums), and for each possible number of segments m.
        for i in range(1, n + 1):
            # m can be at most i segments (each segment non-empty).
            for m in range(1, i + 1):
                # We try all possible previous partition points; j from m-1 to i-1.
                # j is the index that marks the end of the previous partition, so current segment covers indices j to i-1.
                for j in range(m - 1, i):
                    # cost of the new (m-th) subarray from j to i-1:
                    # Sum of nums for subarray + additional k * m: (prefix sum of nums for first i) - (prefix sum for first j) 
                    # then plus k * m, but we can write that as (preN[i] - preN[j] + k*m)
                    # multiplied by the sum of costs in that subarray: preC[i] - preC[j]
                    cost_sub = ( (preN[i] - preN[j]) + k * m ) * (preC[i] - preC[j])
                    dp[i][m] = min(dp[i][m], dp[j][m-1] + cost_sub)
        
        # The answer is the minimal total cost for partitioning the entire array into any valid number of segments (from 1 to n)
        ans = min(dp[n][m] for m in range(1, n+1))
        return ans