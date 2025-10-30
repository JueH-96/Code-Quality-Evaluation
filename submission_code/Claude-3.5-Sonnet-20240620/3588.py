class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Initialize dp array
        dp = [[[0] * 3 for _ in range(3)] for _ in range(n + 1)]
        
        # Base case: Bob can start with any creature
        for j in range(3):
            dp[0][j][0] = 1
        
        # Define the winning relationships
        wins = {'F': 'E', 'W': 'F', 'E': 'W'}
        
        # Fill the dp array
        for i in range(1, n + 1):
            alice_move = s[i-1]
            for last_move in range(3):
                for bob_score in range(i + 1):
                    if bob_score > i // 2:  # Bob is already winning
                        continue
                    
                    for bob_move in range(3):
                        if bob_move == last_move:  # Bob can't make the same move twice
                            continue
                        
                        bob_wins = wins[alice_move] == 'FWE'[bob_move]
                        alice_wins = wins['FWE'[bob_move]] == alice_move
                        
                        if bob_wins:
                            dp[i][bob_move][bob_score + 1] += dp[i-1][last_move][bob_score]
                        elif not alice_wins:
                            dp[i][bob_move][bob_score] += dp[i-1][last_move][bob_score]
                        
                        dp[i][bob_move][bob_score] %= MOD
                        dp[i][bob_move][bob_score + 1] %= MOD
        
        # Sum up all winning sequences for Bob
        result = 0
        for j in range(3):
            for k in range((n + 1) // 2 + 1, n + 1):
                result += dp[n][j][k]
                result %= MOD
        
        return result