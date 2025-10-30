class Solution:
    def countWinningSequences(self, s: str) -> int:
        """
        Calculates the number of distinct sequences Bob can use to beat Alice.

        This problem is solved using dynamic programming with space optimization.
        The state at each round `i` depends on Bob's move in that round and the
        accumulated score difference.

        Let dp[j][k] be the number of ways Bob can play up to a certain round,
        ending with move `j` and having a score difference of `k`.
        """
        n = len(s)
        MOD = 10**9 + 7

        moves = {'F': 0, 'W': 1, 'E': 2}
        s_numeric = [moves[c] for c in s]

        def get_score(bobs_move: int, alices_move: int) -> int:
            if bobs_move == alices_move:
                return 0
            # F(0) beats E(2), W(1) beats F(0), E(2) beats W(1)
            # This corresponds to bobs_move == (alices_move + 1) % 3
            if bobs_move == (alices_move + 1) % 3:
                return 1  # Bob wins
            else:
                return -1  # Alice wins

        # dp[last_move][diff + n]
        # Using space optimization, dp stores the state of the previous round.
        dp = [[0] * (2 * n + 1) for _ in range(3)]

        # --- Base Case: After round 1 (index 0) ---
        alice_move_0 = s_numeric[0]
        for b_curr in range(3):
            score = get_score(b_curr, alice_move_0)
            dp[b_curr][score + n] = 1

        # --- DP Iteration: For rounds 2 to n (indices 1 to n-1) ---
        # The loop variable `i_round` is the 0-indexed round number.
        for i_round in range(1, n):
            alice_move = s_numeric[i_round]
            
            new_dp = [[0] * (2 * n + 1) for _ in range(3)]

            # Precompute sum over previous moves for each possible previous difference.
            # After i_round rounds (i.e., rounds 0 to i_round-1), the max absolute difference is i_round.
            num_rounds_played = i_round
            total_prev_ways = [0] * (2 * n + 1)
            for prev_diff_idx in range(n - num_rounds_played, n + num_rounds_played + 1):
                total = 0
                for b_prev in range(3):
                    total = (total + dp[b_prev][prev_diff_idx])
                total_prev_ways[prev_diff_idx] = total % MOD

            # Fill new_dp for the current round
            for b_curr in range(3):
                score_change = get_score(b_curr, alice_move)
                
                # Iterate over possible previous differences.
                for prev_diff in range(-num_rounds_played, num_rounds_played + 1):
                    prev_diff_idx = prev_diff + n
                    
                    # Number of ways to reach prev_diff ending with a move other than b_curr
                    ways_from_others = (total_prev_ways[prev_diff_idx] - dp[b_curr][prev_diff_idx] + MOD) % MOD
                    
                    if ways_from_others > 0:
                        curr_diff = prev_diff + score_change
                        curr_diff_idx = curr_diff + n
                        new_dp[b_curr][curr_diff_idx] = (new_dp[b_curr][curr_diff_idx] + ways_from_others) % MOD
            dp = new_dp

        # --- Final Calculation ---
        ans = 0
        # Sum over all possible last moves
        for b_last in range(3):
            # Sum over all winning differences (diff > 0)
            for diff in range(1, n + 1):
                diff_idx = diff + n
                ans = (ans + dp[b_last][diff_idx]) % MOD
        
        return ans