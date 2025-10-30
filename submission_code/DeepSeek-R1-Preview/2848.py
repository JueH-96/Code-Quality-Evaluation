class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        # Precompute compatible matrix
        compatible = [[False for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                    compatible[i][j] = True
        max_mask = 1 << n
        dp = [[0] * n for _ in range(max_mask)]
        # Initialize for masks with single element
        for i in range(n):
            dp[1 << i][i] = 1
        # Process all masks
        for mask in range(max_mask):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                # Try adding each possible next element
                for next in range(n):
                    if mask & (1 << next):
                        continue  # next already in mask
                    if compatible[last][next]:
                        new_mask = mask | (1 << next)
                        dp[new_mask][next] = (dp[new_mask][next] + dp[mask][last]) % MOD
        # Sum all possibilities for full mask
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][i] for i in range(n)) % MOD