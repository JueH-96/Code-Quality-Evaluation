class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [[0] * (1 << n) for _ in range(n)]
        
        for i in range(n):
            dp[i][1 << i] = 1
        
        for mask in range(1 << n):
            for i in range(n):
                if mask & (1 << i):
                    for j in range(n):
                        if not (mask & (1 << j)) and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                            dp[j][mask | (1 << j)] = (dp[j][mask | (1 << j)] + dp[i][mask]) % MOD
        
        return sum(dp[i][(1 << n) - 1] for i in range(n)) % MOD