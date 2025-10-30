from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute adjacency: ok[i][j] = True if nums[i] and nums[j] satisfy divisibility
        ok = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        ok[i][j] = True
        
        # dp[mask][last] = number of ways to build permutation of elements in mask ending at index last
        N = 1 << n
        dp = [[0]*n for _ in range(N)]
        
        # Base cases: single-element permutations
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Fill dp
        for mask in range(N):
            for last in range(n):
                cnt = dp[mask][last]
                if cnt == 0:
                    continue
                # Try to append a new element 'nxt'
                for nxt in range(n):
                    if (mask >> nxt) & 1 == 0 and ok[last][nxt]:
                        dp[mask | (1 << nxt)][nxt] = (dp[mask | (1 << nxt)][nxt] + cnt) % MOD
        
        full_mask = N - 1
        # Sum over all permutations that use all elements
        result = sum(dp[full_mask][i] for i in range(n)) % MOD
        return result