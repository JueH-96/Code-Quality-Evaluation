class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute the compatibility matrix
        compat = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    compat[i][j] = True
        
        # Initialize DP table. dp[mask][last] = number of ways
        dp = [[0] * n for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Process all masks
        for mask in range(1, 1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                current = dp[mask][last]
                if current == 0:
                    continue
                # Try all possible next elements
                for j in range(n):
                    if not (mask & (1 << j)) and compat[last][j]:
                        new_mask = mask | (1 << j)
                        dp[new_mask][j] = (dp[new_mask][j] + current) % MOD
        
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][i] for i in range(n)) % MOD