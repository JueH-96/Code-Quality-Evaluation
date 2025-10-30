class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        if n < 4:
            return 0
        
        # dp[i][a][b][c] will store the number of ways to form a string of length i
        # with a 'l's, b 'e's, and c 't's such that a <= 1, b <= 2, c <= 1
        dp = [[[[0] * 2 for _ in range(3)] for _ in range(2)] for _ in range(n + 1)]
        
        # Base case: empty string
        dp[0][0][0][0] = 1
        
        for i in range(1, n + 1):
            for a in range(2):
                for b in range(3):
                    for c in range(2):
                        # Add a character other than 'l', 'e', 't'
                        dp[i][a][b][c] = (dp[i][a][b][c] + dp[i - 1][a][b][c] * 23) % MOD
                        
                        # Add 'l'
                        if a < 1:
                            dp[i][a + 1][b][c] = (dp[i][a + 1][b][c] + dp[i - 1][a][b][c]) % MOD
                        
                        # Add 'e'
                        if b < 2:
                            dp[i][a][b + 1][c] = (dp[i][a][b + 1][c] + dp[i - 1][a][b][c]) % MOD
                        
                        # Add 't'
                        if c < 1:
                            dp[i][a][b][c + 1] = (dp[i][a][b][c + 1] + dp[i - 1][a][b][c]) % MOD
        
        # Sum up all the ways to form a string of length n with at least one 'l', two 'e's, and one 't'
        result = 0
        for a in range(1, 2):
            for b in range(2, 3):
                for c in range(1, 2):
                    result = (result + dp[n][a][b][c]) % MOD
        
        return result