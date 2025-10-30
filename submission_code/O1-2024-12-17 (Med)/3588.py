class Solution:
    def countWinningSequences(self, s: str) -> int:
        """
        We want to count how many distinct sequences Bob can summon (of the same length as Alice's),
        subject to Bob never using the same creature in consecutive rounds, such that Bob's total
        points exceed Alice's total points.

        Scoring rule per round (Alice's move A, Bob's move B):
          - If A == B, no one gets a point (differenceDelta = 0).
          - F vs E => F gets a point (differenceDelta = -1 if Alice=F, +1 if Bob=F).
          - W vs F => W gets a point
          - E vs W => E gets a point

        In condensed form (differenceDelta = BobPoints - AlicePoints for that round):
          (A, B) -> differenceDelta:
            (F, F) = 0
            (F, W) = +1
            (F, E) = -1
            (W, F) = -1
            (W, W) = 0
            (W, E) = +1
            (E, F) = +1
            (E, W) = -1
            (E, E) = 0

        Let n = len(s). We'll map F->0, W->1, E->2 for Alice's moves and Bob's moves.
        We'll use dynamic programming over:
          i = current round index (1-based),
          last_bob in {0,1,2,3} where 3 is a sentinel meaning "no previous move" (for i=0),
          diff in range [-n, n] offset by +n when stored (so array index is diff + n).
        
        dp[i][last_bob][diff + n] = number of ways to reach round i with Bob's last move = last_bob
                                    and (BobScore - AliceScore) = diff.
        
        We'll use a rolling 2D approach to reduce memory:
          dp_older[last_bob][diff_offset], dp_newer[last_bob][diff_offset]
        
        Transition:
          For each last_bob, each diff_offset, we pick a new bob_move != last_bob (unless last_bob=3),
          compute new_diff_offset = diff_offset + differenceDelta[alice_move][bob_move].
          We only keep it if 0 <= new_diff_offset <= 2n.
        
        At the end, sum up dp_older[bob_move][diff_offset] for bob_move in [0,1,2] and
        diff_offset > n (which means diff > 0). Return that sum modulo 1e9+7.
        """

        MOD = 10**9 + 7
        n = len(s)

        # Map each char in s to 0,1,2 (F, W, E)
        char_map = {'F': 0, 'W': 1, 'E': 2}
        alice_moves = [char_map[ch] for ch in s]

        # differenceDelta[a][b] = how (BobPoints - AlicePoints) changes
        # when Alice=a, Bob=b in one round
        differenceDelta = [
            [0,  1, -1],  # A=F=0, B=F=0->0, B=W=1->+1, B=E=2->-1
            [-1, 0,  1],  # A=W=1, B=F=0->-1, B=W=1->0,  B=E=2->+1
            [1, -1,  0]   # A=E=2, B=F=0->+1, B=W=1->-1, B=E=2->0
        ]

        # dp_older[last_bob][diff_offset]
        # We'll have an extra row for last_bob = 3 (sentinel, meaning no move chosen yet)
        offset = n
        dp_older = [[0]*(2*n + 1) for _ in range(4)]
        dp_newer = [[0]*(2*n + 1) for _ in range(4)]

        # Initially, for i=0, we haven't chosen any moves, difference=0 => dp[3][offset] = 1
        dp_older[3][offset] = 1

        for i in range(n):
            a_move = alice_moves[i]
            # Clear dp_newer
            for row in dp_newer:
                for j in range(len(row)):
                    row[j] = 0

            for old_bob in range(4):
                for diff_offset_val in range(2*n + 1):
                    ways = dp_older[old_bob][diff_offset_val]
                    if ways == 0:
                        continue
                    for new_bob in range(3):
                        # Bob cannot repeat the same creature in consecutive rounds
                        # except old_bob=3 (starting sentinel) where we can choose any
                        if old_bob != 3 and new_bob == old_bob:
                            continue
                        d_change = differenceDelta[a_move][new_bob]
                        new_diff_offset = diff_offset_val + d_change
                        if 0 <= new_diff_offset <= 2*n:
                            dp_newer[new_bob][new_diff_offset] = (dp_newer[new_bob][new_diff_offset] + ways) % MOD

            # Swap
            dp_older, dp_newer = dp_newer, dp_older

        # Sum all dp_older[bob_move][diff_offset] for diff > 0 => diff_offset > n
        result = 0
        for bob_move in range(3):
            for diff_offset_val in range(n+1, 2*n+1):
                result = (result + dp_older[bob_move][diff_offset_val]) % MOD

        return result