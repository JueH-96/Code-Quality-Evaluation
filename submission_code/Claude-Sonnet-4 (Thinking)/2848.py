class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        # Precompute adjacency matrix
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i][j] = nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0
        
        # Memoization
        memo = {}
        
        def dp(mask, last):
            if mask == (1 << n) - 1:  # All elements used
                return 1
            
            if (mask, last) in memo:
                return memo[(mask, last)]
            
            result = 0
            for next_idx in range(n):
                if not (mask & (1 << next_idx)):  # Element not used yet
                    if last == -1 or adj[last][next_idx]:
                        result = (result + dp(mask | (1 << next_idx), next_idx)) % MOD
            
            memo[(mask, last)] = result
            return result
        
        return dp(0, -1)