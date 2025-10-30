class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][j][m] = number of subsequences of nums[0:i] with sum j and size m
        dp = [[[0 for _ in range(n+1)] for _ in range(k+1)] for _ in range(n+1)]
        
        # Base case: empty subsequence
        for i in range(n+1):
            dp[i][0][0] = 1
        
        for i in range(1, n+1):
            for j in range(k+1):
                for m in range(i+1):
                    # Don't include nums[i-1]
                    dp[i][j][m] = dp[i-1][j][m]
                    # Include nums[i-1]
                    if j >= nums[i-1] and m >= 1:
                        dp[i][j][m] = (dp[i][j][m] + dp[i-1][j-nums[i-1]][m-1]) % MOD
        
        # Precompute powers of 2
        pow2 = [1]
        for i in range(1, n+1):
            pow2.append((pow2[-1] * 2) % MOD)
        
        # Calculate the answer
        result = 0
        for m in range(n+1):
            result = (result + dp[n][k][m] * pow2[n-m]) % MOD
        
        return result