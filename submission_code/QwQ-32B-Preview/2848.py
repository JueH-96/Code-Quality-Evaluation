class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        import itertools
        from typing import List
        
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute compatibility list
        compat = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        compat[i].append(j)
        
        # Initialize DP table
        dp = [[0 for _ in range(n)] for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 1  # Starting with each number alone
        
        # Fill DP table
        for mask in range(1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue  # last is not in mask
                for neighbor in compat[last]:
                    if mask & (1 << neighbor):
                        continue  # neighbor already in mask
                    new_mask = mask | (1 << neighbor)
                    dp[new_mask][neighbor] += dp[mask][last]
                    dp[new_mask][neighbor] %= MOD
        
        # Sum up all dp[full_mask][last]
        full_mask = (1 << n) - 1
        result = 0
        for i in range(n):
            result += dp[full_mask][i]
            result %= MOD
        return result