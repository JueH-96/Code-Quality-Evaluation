class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        MOD = 10**9 + 7
        DIFF_OFFSET = n # Offset to handle negative differences
        # Max possible total difference after n rounds is n (Bob wins all by 1 point), min is -n (Alice wins all by 1 point).
        # The range of total difference is [-n, n].
        # After adding DIFF_OFFSET = n, the shifted difference range is [0, 2n].

        char_to_idx = {'F': 0, 'W': 1, 'E': 2}
        # score_diff_table[alice_idx][bob_idx] = bob_score - alice_score
        # Bob gets +1: (E, F), (F, W), (W, E)
        # Alice gets +1 (-1 for Bob): (F, E), (W, F), (E, W)
        # Draw (0 for Bob): (F, F), (W, W), (E, E)
        score_diff_table = [
            [0, 1, -1],  # Alice 'F' vs Bob 'F', 'W', 'E'. Bob wins vs W (+1), Loses vs E (-1), Draws vs F (0)
            [-1, 0, 1],  # Alice 'W' vs Bob 'F', 'W', 'E'. Bob loses vs F (-1), Draws vs W (0), Wins vs E (+1)
            [1, -1, 0]   # Alice 'E' vs Bob 'F', 'W', 'E'. Bob wins vs F (+1), Loses vs W (-1), Draws vs E (0)
        ]

        # dp[round_idx % 2][bob_move_idx_in_round][shifted_diff]
        # round_idx from 0 to n-1
        # bob_move_idx_in_round: 0 ('F'), 1 ('W'), 2 ('E')
        # shifted_diff: total_diff_after_round + DIFF_OFFSET, range [0, 2*n]
        # Using 2 layers for space optimization: dp[0] and dp[1]
        dp = [[[0 for _ in range(2 * n + 1)] for _ in range(3)] for _ in range(2)]

        # Base case: Round 0 (index 0)
        a0_idx = char_to_idx[s[0]]
        for b0_idx in range(3): # Bob's possible moves in round 0 ('F', 'W', 'E')
            diff0 = score_diff_table[a0_idx][b0_idx]
            shifted_diff0 = diff0 + DIFF_OFFSET
            # There is 1 way to reach this state (by choosing this b0_idx as the first move)
            if 0 <= shifted_diff0 <= 2 * n: # Ensure the initial shifted diff is within bounds
                 dp[0][b0_idx][shifted_diff0] = 1

        # DP transitions: Round i from 1 to n-1
        for i in range(1, n):
            curr = i % 2 # Current DP layer (for round i)
            prev = (i - 1) % 2 # Previous DP layer (for round i-1)

            # Reset the current DP layer to 0 before filling it with counts for round i
            dp[curr] = [[0 for _ in range(2 * n + 1)] for _ in range(3)]

            ai_idx = char_to_idx[s[i]] # Alice's move in round i

            for bi_idx in range(3): # Bob's possible move in round i ('F', 'W', 'E')
                score_diff_i = score_diff_table[ai_idx][bi_idx] # Score difference in round i if Bob plays bi_idx

                for prev_b_idx in range(3): # Bob's possible move in round i-1 ('F', 'W', 'E')
                    # Constraint: Bob cannot play the same creature in consecutive rounds
                    if bi_idx != prev_b_idx:
                        # Iterate through all possible previous shifted differences after round i-1
                        for prev_shifted_diff in range(2 * n + 1):
                             # Only consider states from the previous round that are reachable (count > 0)
                            if dp[prev][prev_b_idx][prev_shifted_diff] > 0:
                                # Calculate the new shifted difference after round i
                                # New_total_diff = Prev_total_diff + Current_round_diff
                                # New_shifted_diff - n = (Prev_shifted_diff - n) + Current_round_diff
                                # New_shifted_diff = Prev_shifted_diff + Current_round_diff
                                current_shifted_diff = prev_shifted_diff + score_diff_i

                                # The resulting shifted_diff must be within the valid range [0, 2n]
                                # which corresponds to total difference [-n, n].
                                # The difference after round i is mathematically bounded by [-(i+1), i+1].
                                # The shifted difference is bounded by [n-(i+1), n+(i+1)].
                                # These ranges are always within [0, 2n] for i < n.
                                # However, explicitly checking the bounds handles potential edge cases gracefully.
                                if 0 <= current_shifted_diff <= 2 * n:
                                     # Add the number of ways to reach the previous state (dp[prev][prev_b_idx][prev_shifted_diff])
                                     # to the number of ways to reach the current state (dp[curr][bi_idx][current_shifted_diff]).
                                     # Apply modulo operation at each addition to prevent overflow.
                                     dp[curr][bi_idx][current_shifted_diff] = (dp[curr][bi_idx][current_shifted_diff] + dp[prev][prev_b_idx][prev_shifted_diff]) % MOD


        # Final result: Sum up counts for all sequences after n rounds (index n-1) where Bob wins (total diff > 0)
        # The counts after n rounds are stored in dp[(n-1)%2]
        final_round_dp_layer = (n - 1) % 2
        total_winning_sequences = 0

        for b_idx in range(3): # Sum over all possible last moves for Bob ('F', 'W', 'E')
            # Sum up counts for shifted_diff values corresponding to total_diff > 0
            # total_diff = shifted_diff - DIFF_OFFSET
            # total_diff > 0 implies shifted_diff - n > 0, so shifted_diff > n
            # The shifted differences range from 0 to 2n. We need to sum from n + 1 up to 2n.
            # The range function is inclusive start, exclusive stop. So it should be range(n + 1, 2*n + 1).
            for shifted_diff in range(DIFF_OFFSET + 1, 2 * n + 1):
                 total_winning_sequences = (total_winning_sequences + dp[final_round_dp_layer][b_idx][shifted_diff]) % MOD

        return total_winning_sequences