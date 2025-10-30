class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        MOD = 10**9 + 7
        
        # Calculate score difference for a round (Bob's score - Alice's score)
        def get_score_diff(alice_move, bob_move):
            if alice_move == bob_move:
                return 0
            
            # Bob scores a point
            if ((alice_move == 'F' and bob_move == 'W') or 
                (alice_move == 'W' and bob_move == 'E') or 
                (alice_move == 'E' and bob_move == 'F')):
                return 1
            
            # Alice scores a point
            return -1
        
        # Define dp[i][prev][score] = number of valid sequences such that:
        # - we've played i rounds
        # - Bob's last move was 'prev'
        # - the score difference (Bob - Alice) is 'score'
        
        moves = ['F', 'W', 'E']
        
        # Initialize dp array
        dp = [[[0 for _ in range(2*n+1)] for _ in range(3)] for _ in range(n+1)]
        
        # Base case: after first round
        for move_idx, move in enumerate(moves):
            score_diff = get_score_diff(s[0], move)
            dp[1][move_idx][n + score_diff] = 1
        
        # Fill dp table
        for i in range(2, n+1):
            for move_idx, move in enumerate(moves):
                for prev_idx, prev in enumerate(moves):
                    if move_idx != prev_idx:  # Bob can't repeat moves
                        score_diff = get_score_diff(s[i-1], move)
                        for old_score in range(2*n+1):
                            if dp[i-1][prev_idx][old_score] > 0:
                                new_score = old_score + score_diff
                                if 0 <= new_score < 2*n+1:
                                    dp[i][move_idx][new_score] = (dp[i][move_idx][new_score] + 
                                                                 dp[i-1][prev_idx][old_score]) % MOD
        
        # Count sequences where Bob wins (positive score difference)
        result = 0
        for move_idx in range(3):
            for score in range(n+1, 2*n+1):  # Scores > 0
                result = (result + dp[n][move_idx][score]) % MOD
        
        return result