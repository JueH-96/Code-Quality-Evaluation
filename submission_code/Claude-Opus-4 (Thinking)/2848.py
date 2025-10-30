class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        # dp[mask][last] = number of special permutations using elements in mask and ending with last
        dp = [[0] * n for _ in range(1 << n)]
        
        # Base case: single element permutations
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Fill dp table
        for mask in range(1, 1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if dp[mask][last] == 0:
                    continue
                
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    
                    if nums[last] % nums[nxt] == 0 or nums[nxt] % nums[last] == 0:
                        new_mask = mask | (1 << nxt)
                        dp[new_mask][nxt] = (dp[new_mask][nxt] + dp[mask][last]) % MOD
        
        # Sum all valid permutations
        full_mask = (1 << n) - 1
        result = 0
        for i in range(n):
            result = (result + dp[full_mask][i]) % MOD
        
        return result