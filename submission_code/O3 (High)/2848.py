from typing import List

MOD = 10**9 + 7

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        # Pre-compute which indices can be adjacent in either order.
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                    adj[i].append(j)

        full_mask = 1 << n
        # dp[mask][last]  = number of ways to pick indices in 'mask'
        #                   such that the last picked index is 'last'
        dp = [[0] * n for _ in range(full_mask)]
        for i in range(n):
            dp[1 << i][i] = 1

        for mask in range(full_mask):
            for last in range(n):
                if dp[mask][last] == 0:
                    continue
                cur_val = dp[mask][last]
                for nxt in adj[last]:
                    if mask & (1 << nxt):
                        continue           # nxt already used
                    new_mask = mask | (1 << nxt)
                    dp[new_mask][nxt] = (dp[new_mask][nxt] + cur_val) % MOD

        return sum(dp[full_mask - 1]) % MOD