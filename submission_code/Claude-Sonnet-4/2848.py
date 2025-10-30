class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        # Precompute which pairs can be adjacent
        can_follow = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        can_follow[i][j] = True
        
        # dp[mask][last] = number of ways to arrange numbers in mask ending with last
        dp = {}
        
        def solve(mask, last):
            if mask == (1 << n) - 1:  # All numbers used
                return 1
            
            if (mask, last) in dp:
                return dp[(mask, last)]
            
            result = 0
            for next_idx in range(n):
                if mask & (1 << next_idx):  # Already used
                    continue
                
                if last == -1 or can_follow[last][next_idx]:
                    result = (result + solve(mask | (1 << next_idx), next_idx)) % MOD
            
            dp[(mask, last)] = result
            return result
        
        return solve(0, -1)