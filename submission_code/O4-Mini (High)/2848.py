from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Build adjacency: can go from i to j if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    adj[i].append(j)
        
        # dp[mask][i] = # of ways to pick a permutation of the elements in 'mask'
        # that ends at index i
        full_mask = (1 << n) - 1
        dp = [[0] * n for _ in range(1 << n)]
        
        # Initialize: single-element permutations
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Fill DP
        for mask in range(1 << n):
            for last in range(n):
                ways = dp[mask][last]
                if ways == 0:
                    continue
                # try to append a new element 'nxt'
                for nxt in adj[last]:
                    if not (mask & (1 << nxt)):
                        new_mask = mask | (1 << nxt)
                        dp[new_mask][nxt] = (dp[new_mask][nxt] + ways) % MOD
        
        # Sum over all permutations that use all elements
        ans = sum(dp[full_mask][i] for i in range(n)) % MOD
        return ans