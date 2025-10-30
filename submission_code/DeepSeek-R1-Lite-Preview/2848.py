from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        adj[i].append(j)
        
        # Initialize DP table
        dp = [[0] * n for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Iterate over all masks
        for mask in range(1 << n):
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                for j in adj[i]:
                    if mask & (1 << j):
                        continue
                    dp[mask | (1 << j)][j] = (dp[mask | (1 << j)][j] + dp[mask][i]) % MOD
        
        # Sum up the results for full mask
        full_mask = (1 << n) - 1
        result = 0
        for i in range(n):
            result = (result + dp[full_mask][i]) % MOD
        return result