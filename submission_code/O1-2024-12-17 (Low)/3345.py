class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        # Precompute powers of 2 up to n
        pow2 = [1] * (n+1)
        for i in range(1, n+1):
            pow2[i] = (pow2[i-1] * 2) % MOD

        # dp[i][j][l] = number of subsequences (of length l) among the first i elements with sum j
        # Dimensions: (n+1) x (k+1) x (n+1)
        dp = [[[0]*(n+1) for _ in range(k+1)] for _ in range(n+1)]
        
        # Base case: dp[0][0][0] = 1 (one empty subsequence of length 0, sum 0)
        dp[0][0][0] = 1

        for i in range(1, n+1):
            val = nums[i-1]
            for j in range(k+1):
                for l in range(i+1):
                    # Case 1: not taking nums[i-1]
                    dp[i][j][l] += dp[i-1][j][l]
                    dp[i][j][l] %= MOD
                    # Case 2: take nums[i-1] (if it doesn't exceed j and l > 0)
                    if l > 0 and j >= val:
                        dp[i][j][l] += dp[i-1][j-val][l-1]
                        dp[i][j][l] %= MOD

        # Now sum over all lengths l from 1..n of dp[n][k][l] * 2^(n-l)
        ans = 0
        for l in range(1, n+1):
            ans += dp[n][k][l] * pow2[n - l]
            ans %= MOD

        return ans % MOD