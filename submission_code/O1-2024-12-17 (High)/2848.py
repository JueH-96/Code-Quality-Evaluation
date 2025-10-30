class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute adjacency: adj[i][j] = True if nums[i] and nums[j] satisfy
        # that one divides the other.
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        adj[i][j] = True
        
        # dp[mask][j] will store the number of ways to form a valid arrangement
        # (subset indicated by mask) that ends with element j.
        dp = [[0] * n for _ in range(1 << n)]
        
        # Initialize dp for single-element subsets
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Build up the dp table
        for mask in range(1 << n):
            for j in range(n):
                if dp[mask][j] == 0:
                    continue
                # Try to pick a new element k not in 'mask'
                # and check if it can follow j according to adjacency
                for k in range(n):
                    if (mask & (1 << k)) == 0 and adj[j][k]:
                        dp[mask | (1 << k)][k] = (dp[mask | (1 << k)][k] + dp[mask][j]) % MOD
        
        # Sum the ways over all possible ending elements with all elements used
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][j] for j in range(n)) % MOD