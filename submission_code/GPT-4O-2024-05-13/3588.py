class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j] will store the number of ways Bob can win up to the i-th round
        # with Bob's last move being j where j is 0 for 'F', 1 for 'W', 2 for 'E'
        dp = [[0] * 3 for _ in range(len(s) + 1)]
        
        # Initial state, no rounds played, no last move
        dp[0][0] = dp[0][1] = dp[0][2] = 1
        
        for i in range(1, len(s) + 1):
            for j in range(3):
                for k in range(3):
                    if j != k:
                        if (s[i-1] == 'F' and k == 1) or (s[i-1] == 'W' and k == 2) or (s[i-1] == 'E' and k == 0):
                            dp[i][k] = (dp[i][k] + dp[i-1][j]) % MOD
        
        # Sum up all valid sequences where Bob wins
        result = sum(dp[len(s)][j] for j in range(3)) % MOD
        
        return result