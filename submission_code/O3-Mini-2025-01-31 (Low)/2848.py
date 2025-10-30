from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        
        # Precompute valid transitions
        valid = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    valid[i][j] = True
        
        # dp[mask][i]: number of valid sequences using elements of mask ending with index i
        dp = [[0] * n for _ in range(1 << n)]
        
        # Starting point for each index
        for i in range(n):
            dp[1 << i][i] = 1
        
        # DP over bitmask
        for mask in range(1 << n):
            for i in range(n):
                if not (mask & (1 << i)):  # i is not in mask, skip
                    continue
                # Try to add a new j not in mask
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    if valid[i][j]:
                        new_mask = mask | (1 << j)
                        dp[new_mask][j] = (dp[new_mask][j] + dp[mask][i]) % mod
        
        # Answer is sum of dp[(1<<n)-1][i] for all end indices i
        full_mask = (1 << n) - 1
        result = sum(dp[full_mask][i] for i in range(n)) % mod
        
        return result