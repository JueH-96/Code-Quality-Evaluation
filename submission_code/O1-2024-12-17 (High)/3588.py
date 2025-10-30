class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)

        # Map each character to an integer for convenience:
        # 'F' -> 0, 'W' -> 1, 'E' -> 2
        def char_to_int(c):
            if c == 'F':
                return 0
            elif c == 'W':
                return 1
            else:
                return 2

        # Precompute alice's moves in integer form
        alice_moves = [char_to_int(c) for c in s]

        # diff_mat[bob_move][alice_move] = (bob_points - alice_points) in a single round
        #  bob_move, alice_move ∈ {0=F, 1=W, 2=E}
        diff_mat = [
            [0, -1,  1],  # Bob=F vs Alice=F/W/E
            [1,  0, -1],  # Bob=W vs Alice=F/W/E
            [-1, 1,  0]   # Bob=E vs Alice=F/W/E
        ]

        # We use a DP approach where:
        # dp[move][diff_index] = number of ways for Bob up to the current round
        #   ending with bob_move = move, and the overall score difference
        #   (bob_score - alice_score) = (diff_index - offset).
        # offset = n, so diff_index goes from 0 to 2*n, representing score differences -n..n.
        offset = n

        # dp1 and dp2 will each be 3 x (2n+1) arrays.
        # dp1 corresponds to the distribution after processing i-1 rounds,
        # dp2 corresponds to the next distribution after round i.
        dp1 = [[0]*(2*n + 1) for _ in range(3)]
        dp2 = [[0]*(2*n + 1) for _ in range(3)]

        # Initialize dp for the first round (i=0).
        # Bob can choose any of F(0), W(1), E(2).
        # The new difference index is offset + diff_mat[bob_move][alice_moves[0]].
        first_alice_move = alice_moves[0]
        for bob_move in range(3):
            new_diff_index = offset + diff_mat[bob_move][first_alice_move]
            if 0 <= new_diff_index <= 2*n:
                dp1[bob_move][new_diff_index] = (dp1[bob_move][new_diff_index] + 1) % MOD

        # Fill dp for rounds i = 1..n-1
        for i in range(1, n):
            # Clear dp2 for the next iteration
            for m in range(3):
                for d in range(2*n + 1):
                    dp2[m][d] = 0

            a_move = alice_moves[i]
            for last_move in range(3):
                row = dp1[last_move]
                for diff_index in range(2*n + 1):
                    ways = row[diff_index]
                    if ways == 0:
                        continue
                    # Bob's next move must differ from last_move
                    for next_move in range(3):
                        if next_move == last_move:
                            continue
                        nd = diff_index + diff_mat[next_move][a_move]
                        if 0 <= nd <= 2*n:
                            dp2[next_move][nd] = (dp2[next_move][nd] + ways) % MOD

            # Swap dp1 and dp2
            dp1, dp2 = dp2, dp1

        # Sum all ways where final (bob_score - alice_score) > 0
        # i.e., diff_index > offset
        result = 0
        for bob_move in range(3):
            for diff_index in range(offset+1, 2*n+1):
                result = (result + dp1[bob_move][diff_index]) % MOD

        return result