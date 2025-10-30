class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [ [0] * (k+1) for _ in range(n+1) ]
        dp[0][0] = 1
        
        for i in range(1, n+1):
            num = nums[i-1]
            for j in range(k+1):
                dp[i][j] = dp[i-1][j]
                if j >= num:
                    dp[i][j] = (dp[i][j] + dp[i-1][j - num]) % MOD
        
        total = 0
        for i in range(n+1):
            if dp[i][k] != 0:
                total = (total + dp[i][k] * (1 << (n - i))) % MOD
        
        return total