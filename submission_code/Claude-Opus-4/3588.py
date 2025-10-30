class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Map characters to indices for easier handling
        char_to_idx = {'F': 0, 'W': 1, 'E': 2}
        alice_moves = [char_to_idx[c] for c in s]
        
        # Function to determine the result of a round
        # Returns 1 if Bob wins, -1 if Alice wins, 0 if tie
        def get_result(alice_move, bob_move):
            if alice_move == bob_move:
                return 0
            # F beats E, W beats F, E beats W
            if (alice_move == 0 and bob_move == 2) or \
               (alice_move == 1 and bob_move == 0) or \
               (alice_move == 2 and bob_move == 1):
                return -1  # Alice wins
            return 1  # Bob wins
        
        # DP: dp[round][last_bob_move][score_diff + offset]
        # score_diff ranges from -n to n, so we use offset n
        offset = n
        dp = [[[0] * (2 * n + 1) for _ in range(3)] for _ in range(n + 1)]
        
        # Base case: round 0, no moves yet
        # We use -1 to indicate no previous move for Bob
        for first_move in range(3):
            result = get_result(alice_moves[0], first_move)
            dp[1][first_move][result + offset] = 1
        
        # Fill the DP table
        for round_idx in range(1, n):
            alice_move = alice_moves[round_idx]
            
            for prev_bob_move in range(3):
                for score_diff in range(-n, n + 1):
                    if dp[round_idx][prev_bob_move][score_diff + offset] == 0:
                        continue
                    
                    for curr_bob_move in range(3):
                        if curr_bob_move == prev_bob_move:
                            continue  # Bob can't repeat moves
                        
                        result = get_result(alice_move, curr_bob_move)
                        new_score_diff = score_diff + result
                        
                        if -n <= new_score_diff <= n:
                            dp[round_idx + 1][curr_bob_move][new_score_diff + offset] += \
                                dp[round_idx][prev_bob_move][score_diff + offset]
                            dp[round_idx + 1][curr_bob_move][new_score_diff + offset] %= MOD
        
        # Count sequences where Bob wins (score_diff > 0)
        count = 0
        for last_move in range(3):
            for score_diff in range(1, n + 1):
                count += dp[n][last_move][score_diff + offset]
                count %= MOD
        
        return count