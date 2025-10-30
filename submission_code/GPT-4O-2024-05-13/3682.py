class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j] will be the number of ways to form an array of length i with j pairs
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        # Base case: there's 1 way to form an array of length 1 with 0 pairs
        dp[1][0] = m
        
        for i in range(2, n + 1):
            for j in range(k + 1):
                # Case 1: The i-th element is different from the (i-1)-th element
                dp[i][j] = dp[i-1][j] * (m - 1) % MOD
                
                # Case 2: The i-th element is the same as the (i-1)-th element
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i-1][j-1]) % MOD
        
        return dp[n][k]

# Example usage:
# sol = Solution()
# print(sol.countGoodArrays(3, 2, 1))  # Output: 4
# print(sol.countGoodArrays(4, 2, 2))  # Output: 6
# print(sol.countGoodArrays(5, 2, 0))  # Output: 2