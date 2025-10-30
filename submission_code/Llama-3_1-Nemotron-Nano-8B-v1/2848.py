from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # Precompute compatibility matrix
        compatible = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    compatible[i][j] = True
        # Initialize DP
        dp = [[0] * n for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 1
        # Iterate over all masks
        for mask in range(1, 1 << n):
            for last in range(n):
                if (mask & (1 << last)) and dp[mask][last]:
                    # Try to add each j not in mask
                    for j in range(n):
                        if not (mask & (1 << j)):
                            if compatible[last][j]:
                                new_mask = mask | (1 << j)
                                dp[new_mask][j] = (dp[new_mask][j] + dp[mask][last]) % MOD
        # Sum all full mask possibilities
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][i] for i in range(n)) % MOD