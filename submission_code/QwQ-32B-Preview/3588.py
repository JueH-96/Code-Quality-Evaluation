class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Define the mapping of choices to indices
        choice_map = {'F': 0, 'W': 1, 'E': 2}
        
        # Initialize the DP table
        # dp[i][j] where i is the round, j is Bob's last choice
        dp = [[0 for _ in range(3)] for _ in range(n)]
        
        # Initialize the first round
        for b in range(3):
            a = choice_map[s[0]]
            if (a == 0 and b == 1):  # F vs W: Bob gets 1 point
                dp[0][b] = 1
            elif (a == 1 and b == 2):  # W vs E: Bob gets 1 point
                dp[0][b] = 1
            elif (a == 2 and b == 0):  # E vs F: Bob gets 1 point
                dp[0][b] = 1
            elif (a == b):  # Same choice: no points
                dp[0][b] = 0
            else:  # Other cases: Alice gets 1 point
                dp[0][b] = -1
            dp[0][b] = max(0, dp[0][b])
        
        # Iterate through each round
        for i in range(1, n):
            a = choice_map[s[i]]
            for b in range(3):
                dp[i][b] = 0
                for prev_b in range(3):
                    if b != prev_b:
                        points_diff = 0
                        if (a == 0 and b == 1):  # F vs W: Bob +1
                            points_diff = 1
                        elif (a == 1 and b == 2):  # W vs E: Bob +1
                            points_diff = 1
                        elif (a == 2 and b == 0):  # E vs F: Bob +1
                            points_diff = 1
                        elif (a == b):  # Same: no points
                            points_diff = 0
                        else:  # Alice gets 1 point: Bob -1
                            points_diff = -1
                        if points_diff + dp[i-1][prev_b] > 0:
                            dp[i][b] = (dp[i][b] + dp[i-1][prev_b]) % MOD
        
        # Sum up the last round's valid sequences
        total = 0
        for b in range(3):
            total = (total + dp[n-1][b]) % MOD
        return total