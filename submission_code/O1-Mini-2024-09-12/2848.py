from typing import List
from functools import lru_cache

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # Precompute adjacency
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    adj[i].append(j)
        @lru_cache(None)
        def dfs(mask, last):
            if mask == (1 << n) -1:
                return 1
            total = 0
            for nei in adj[last]:
                if not mask & (1 << nei):
                    total += dfs(mask | (1 << nei), nei)
                    total %= MOD
            return total
        total = 0
        for i in range(n):
            total += dfs(1 << i, i)
            total %= MOD
        return total