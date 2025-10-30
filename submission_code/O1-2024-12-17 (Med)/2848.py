class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute adjacency: edges[i][j] = True if nums[i] and nums[j] can be adjacent
        edges = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        edges[i][j] = True
        
        # dp[mask][last] = number of special permutations of the elements in 'mask' ending with index 'last'
        dp = [[0]*n for _ in range(1 << n)]
        
        # Base cases: using exactly one element (each index)
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Fill dp table
        for mask in range(1 << n):
            for last in range(n):
                if dp[mask][last] == 0:
                    continue
                # Try to pick the next element
                for nxt in range(n):
                    if (mask & (1 << nxt)) == 0 and edges[last][nxt]:
                        dp[mask | (1 << nxt)][nxt] = (dp[mask | (1 << nxt)][nxt] + dp[mask][last]) % MOD
        
        # Sum up all permutations that use all n elements
        full_mask = (1 << n) - 1
        ans = 0
        for last in range(n):
            ans = (ans + dp[full_mask][last]) % MOD
        
        return ans