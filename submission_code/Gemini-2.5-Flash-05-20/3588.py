class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        MOD = 10**9 + 7
        
        # Map creature chars to integers for array indexing
        # 'F': 0, 'W': 1, 'E': 2
        creatures = ['F', 'W', 'E']
        
        # Define scoring rules: scoring_matrix[Alice_move][Bob_move] = (Alice_points, Bob_points)
        # The key is Alice's creature, the inner dictionary's key is Bob's creature.
        scoring_matrix = {
            'F': {'F': (0, 0), 'W': (0, 1), 'E': (1, 0)}, # Alice: F. Bob: F (0,0), W (W beats F -> 0,1), E (F beats E -> 1,0)
            'W': {'F': (1, 0), 'W': (0, 0), 'E': (0, 1)}, # Alice: W. Bob: F (W beats F -> 1,0), W (0,0), E (E beats W -> 0,1)
            'E': {'F': (0, 1), 'W': (1, 0), 'E': (0, 0)}  # Alice: E. Bob: F (F beats E -> 0,1), W (E beats W -> 1,0), E (0,0)
        }
        
        # OFFSET is used to convert score differences (-n to n) to array indices (0 to 2*n)
        OFFSET = n 
        
        # dp[current_round_idx % 2][last_bob_creature_idx][score_diff_offset]
        # Stores the number of ways to reach this state.
        # Max score difference is N, min is -N. So 2N+1 possible diff values (0 to 2N).
        # We use 2 rows for space optimization: dp[0] for prev state, dp[1] for current state.
        dp = [[[0] * (2 * n + 1) for _ in range(3)] for _ in range(2)]
        
        # Base case: Round 0 (i=0)
        # Alice's first move is s[0]. Bob can play any of F, W, E.
        alice_move_0 = s[0]
        for bob_move_idx_0 in range(3):
            bob_move_0 = creatures[bob_move_idx_0]
            
            alice_pts, bob_pts = scoring_matrix[alice_move_0][bob_move_0]
            score_diff_0 = bob_pts - alice_pts
            
            # The state after round 0 is stored in dp[0]
            dp[0][bob_move_idx_0][score_diff_0 + OFFSET] = 1
            
        # Transitions for subsequent rounds (i from 1 to n-1)
        for i in range(1, n):
            current_dp_idx = i % 2
            prev_dp_idx = (i - 1) % 2
            
            # Clear the current_dp_idx row for this iteration to store new counts
            for j in range(3): # Iterate through bob_move_idx
                for k in range(2 * n + 1): # Iterate through score_diff_offset
                    dp[current_dp_idx][j][k] = 0
            
            alice_current_move = s[i] # Alice's move for the current round
            
            # Iterate through all possible previous states
            for prev_bob_move_idx in range(3):
                for prev_score_diff_offset in range(2 * n + 1):
                    # If there were no ways to reach this previous state, skip
                    if dp[prev_dp_idx][prev_bob_move_idx][prev_score_diff_offset] == 0:
                        continue
                    
                    # Number of ways to reach this specific previous state
                    count_from_prev_state = dp[prev_dp_idx][prev_bob_move_idx][prev_score_diff_offset]
                    
                    # Explore Bob's possible moves for the current round
                    for current_bob_move_idx in range(3):
                        # Constraint: Bob cannot make the same move in consecutive rounds
                        if current_bob_move_idx == prev_bob_move_idx:
                            continue
                            
                        current_bob_move = creatures[current_bob_move_idx]
                        
                        # Calculate points for the current round
                        alice_pts, bob_pts = scoring_matrix[alice_current_move][current_bob_move]
                        round_score_diff = bob_pts - alice_pts
                        
                        # Calculate the new total score difference offset
                        new_score_diff_offset = prev_score_diff_offset + round_score_diff
                        
                        # Add ways to the new state, applying modulo
                        dp[current_dp_idx][current_bob_move_idx][new_score_diff_offset] = \
                            (dp[current_dp_idx][current_bob_move_idx][new_score_diff_offset] + count_from_prev_state) % MOD
                            
        # After processing all n rounds, sum up all states where Bob's score is strictly greater than Alice's
        total_winning_sequences = 0
        final_dp_idx = (n - 1) % 2 # The index of the dp row containing results after n-1 rounds (i.e. all n rounds)
        
        for last_bob_move_idx in range(3):
            for final_score_diff_offset in range(2 * n + 1):
                final_score_diff = final_score_diff_offset - OFFSET
                
                # Bob beats Alice if his total score is strictly greater
                if final_score_diff > 0:
                    total_winning_sequences = (total_winning_sequences + dp[final_dp_idx][last_bob_move_idx][final_score_diff_offset]) % MOD
                    
        return total_winning_sequences