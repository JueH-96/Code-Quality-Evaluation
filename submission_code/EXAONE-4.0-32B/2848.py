class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        
        valid = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                    valid[i][j] = True
        
        total_states = 1 << n
        dp = [[0] * n for _ in range(total_states)]
        
        for i in range(n):
            dp[1 << i][i] = 1
        
        for mask in range(total_states):
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    if valid[i][j]:
                        new_mask = mask | (1 << j)
                        dp[new_mask][j] = (dp[new_mask][j] + dp[mask][i]) % mod
        
        full_mask = (1 << n) - 1
        return sum(dp[full_mask]) % mod