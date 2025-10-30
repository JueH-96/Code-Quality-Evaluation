from bisect import bisect_right
from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        
        # Collect all unique differences between pairs (i, j) where i < j
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(sorted_nums[j] - sorted_nums[i])
        diffs = sorted(diffs)
        diffs.append(10**18)  # Add a dummy value for the last iteration
        
        # Precompute f(d) for each d in diffs
        f = {}
        for d in diffs:
            dp = [[0] * (k + 1) for _ in range(n)]
            for i in range(n):
                dp[i][1] = 1  # Base case: subsequences of length 1
            
            for t in range(2, k + 1):
                # Compute prefix sums for the previous layer (t-1)
                prefix = [0] * n
                prefix[0] = dp[0][t-1]
                for j in range(1, n):
                    prefix[j] = prefix[j-1] + dp[j][t-1]
                
                # Compute dp[i][t] for each i
                for i in range(n):
                    target = sorted_nums[i] - d
                    # Find the largest j < i where sorted_nums[j] <= target
                    idx = bisect_right(sorted_nums, target) - 1
                    idx = min(idx, i - 1)
                    if idx >= 0:
                        dp[i][t] = prefix[idx]
                    else:
                        dp[i][t] = 0
            
            # Sum all dp[i][k] to get f[d]
            total = sum(dp[i][k] for i in range(n))
            f[d] = total % mod
        
        result = 0
        # Iterate through each difference and compute the contribution
        for l in range(len(diffs) - 1):
            current_d = diffs[l]
            next_d = diffs[l + 1]
            contribution = current_d * (f[current_d] - f[next_d])
            result += contribution
        
        return result % mod