class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Function to get score change for Bob
        def get_score_change(alice_char, bob_char):
            if alice_char == bob_char:
                return 0
            if (alice_char == 'F' and bob_char == 'W') or \
               (alice_char == 'W' and bob_char == 'E') or \
               (alice_char == 'E' and bob_char == 'F'):
                return 1  # Bob wins
            else:
                return -1  # Alice wins
        
        # Map characters to indices
        moves = ['F', 'W', 'E']
        
        # DP with rolling array
        # dp[last_bob_move][score_diff + n]
        # last_bob_move: 0='F', 1='W', 2='E'
        # score_diff ranges from -n to n, so we add n as offset
        dp = [[0] * (2 * n + 1) for _ in range(3)]
        
        # Base case: round 0
        for bob_move_idx, bob_move in enumerate(moves):
            score_change = get_score_change(s[0], bob_move)
            dp[bob_move_idx][score_change + n] = 1
        
        # Fill DP for subsequent rounds
        for round_idx in range(1, n):
            alice_move = s[round_idx]
            new_dp = [[0] * (2 * n + 1) for _ in range(3)]
            
            for last_bob_move_idx in range(3):
                for score_diff_offset in range(2 * n + 1):
                    if dp[last_bob_move_idx][score_diff_offset] == 0:
                        continue
                    
                    score_diff = score_diff_offset - n
                    
                    for bob_move_idx, bob_move in enumerate(moves):
                        if bob_move_idx == last_bob_move_idx:
                            continue  # Bob can't play the same move consecutively
                        
                        score_change = get_score_change(alice_move, bob_move)
                        new_score_diff = score_diff + score_change
                        
                        if -n <= new_score_diff <= n:
                            new_score_diff_offset = new_score_diff + n
                            new_dp[bob_move_idx][new_score_diff_offset] = \
                                (new_dp[bob_move_idx][new_score_diff_offset] + 
                                 dp[last_bob_move_idx][score_diff_offset]) % MOD
            
            dp = new_dp
        
        # Count winning sequences (score_diff > 0)
        result = 0
        for last_bob_move_idx in range(3):
            for score_diff in range(1, n + 1):
                score_diff_offset = score_diff + n
                result = (result + dp[last_bob_move_idx][score_diff_offset]) % MOD
        
        return result