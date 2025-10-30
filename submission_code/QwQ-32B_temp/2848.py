class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute compatibility matrix
        compatible = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    a = nums[i]
                    b = nums[j]
                    if a % b == 0 or b % a == 0:
                        compatible[i][j] = True
        
        # Initialize DP table
        dp = [[0] * n for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Iterate over all possible masks
        for mask in range(1 << n):
            for last in range(n):
                if dp[mask][last] == 0:
                    continue
                # Try adding each possible next element
                for j in range(n):
                    if not (mask & (1 << j)):  # j is not in the current mask
                        if compatible[last][j]:
                            new_mask = mask | (1 << j)
                            dp[new_mask][j] = (dp[new_mask][j] + dp[mask][last]) % MOD
        
        # Sum all possibilities for the full mask
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][i] for i in range(n)) % MOD