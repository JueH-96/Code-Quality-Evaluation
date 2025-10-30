class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        MOD = 10 ** 9 + 7
        
        # dp[i][j] represents the number of winning sequences for Bob
        # where Alice's sequence is s[:i] and Bob's last move is j
        # (0 for Fire Dragon, 1 for Water Serpent, 2 for Earth Golem)
        dp = [[0] * 3 for _ in range(n)]
        
        # Initialize the first row of the dp table
        if s[0] == 'F':
            dp[0][0] = 1
        elif s[0] == 'W':
            dp[0][1] = 1
        else:
            dp[0][2] = 1
        
        for i in range(1, n):
            if s[i] == 'F':
                dp[i][0] = (dp[i-1][1] + dp[i-1][2]) % MOD
            elif s[i] == 'W':
                dp[i][1] = (dp[i-1][0] + dp[i-1][2]) % MOD
            else:
                dp[i][2] = (dp[i-1][0] + dp[i-1][1]) % MOD
        
        # Sum up the winning sequences for Bob
        return sum(dp[n-1]) % MOD