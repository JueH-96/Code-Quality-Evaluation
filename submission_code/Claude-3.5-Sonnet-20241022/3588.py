class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # dp[i][last_move][alice_score - bob_score] represents number of ways
        # to achieve score difference with last_move at position i
        # last_move: 0=F, 1=W, 2=E
        # score_diff range: [-n to n]
        dp = [[[0]*21 for _ in range(3)] for _ in range(n+1)]
        
        # Base case: before first move, score diff is 0
        dp[0][0][10] = 1
        dp[0][1][10] = 1
        dp[0][2][10] = 1
        
        # Map moves to numbers
        move_to_num = {'F': 0, 'W': 1, 'E': 2}
        
        # Define who wins in each matchup
        # wins[i][j] = 1 if i beats j, -1 if j beats i, 0 if draw
        wins = [
            [0, -1, 1],  # F vs F, F vs W, F vs E
            [1, 0, -1],  # W vs F, W vs W, W vs E
            [-1, 1, 0]   # E vs F, E vs W, E vs E
        ]
        
        # Fill dp table
        for i in range(n):
            alice_move = move_to_num[s[i]]
            
            for last_bob in range(3):
                for score_diff in range(21):
                    if dp[i][last_bob][score_diff] == 0:
                        continue
                        
                    # Try all possible moves for Bob except last_bob
                    for bob_move in range(3):
                        if bob_move == last_bob:
                            continue
                            
                        # Calculate new score difference
                        result = wins[alice_move][bob_move]
                        new_diff = score_diff
                        if result == 1:  # Alice wins
                            new_diff += 1
                        elif result == -1:  # Bob wins
                            new_diff -= 1
                            
                        if 0 <= new_diff < 21:
                            dp[i+1][bob_move][new_diff] = (
                                dp[i+1][bob_move][new_diff] + 
                                dp[i][last_bob][score_diff]
                            ) % MOD
        
        # Sum all winning sequences for Bob (score_diff < 10)
        result = 0
        for last_move in range(3):
            for score_diff in range(10):
                result = (result + dp[n][last_move][score_diff]) % MOD
                
        return result