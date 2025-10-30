class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # Map characters to integers: F=1, W=2, E=3
        cmap = {'F': 1, 'W': 2, 'E': 3}
        a = [cmap[ch] for ch in s]
        # res[a][b] = (Bob points) - (Alice points) when Alice plays a, Bob plays b
        # Index 1..3 used
        res = [[0]*4 for _ in range(4)]
        # Fill res based on rules
        # F=1, W=2, E=3
        # If (1 vs 3): Fire beats Earth -> whoever played Fire +1
        res[1][3] = -1  # Alice F vs Bob E  => Alice +1 => Bob -1
        res[3][1] =  1  # Alice E vs Bob F  => Bob +1
        # If (2 vs 1): Water beats Fire
        res[2][1] = -1  # Alice W vs Bob F  => Alice +1 => Bob -1
        res[1][2] =  1  # Alice F vs Bob W  => Bob +1
        # If (3 vs 2): Earth beats Water
        res[3][2] = -1  # Alice E vs Bob W  => Alice +1 => Bob -1
        res[2][3] =  1  # Alice W vs Bob E  => Bob +1

        # DP arrays: dp[parity][prev_move][score_offset]
        max_score = 2 * n
        dp = [
            [ [0] * (max_score + 1) for _ in range(4) ],
            [ [0] * (max_score + 1) for _ in range(4) ]
        ]
        mid = n  # offset zero-diff to index mid
        # Initially, before any rounds, prev_move=0 (none), score diff=0
        dp[0][0][mid] = 1
        # Track current valid offset range
        prev_lo = mid
        prev_hi = mid

        for i in range(n):
            j = i & 1
            nj = j ^ 1
            # Clear dp[nj] in the previous offset range
            for pm in range(4):
                row = dp[nj][pm]
                for sc in range(prev_lo, prev_hi + 1):
                    row[sc] = 0
            new_lo = max_score
            new_hi = 0
            ai = a[i]
            # Transition
            for pm in range(4):  # previous Bob move: 0..3
                row = dp[j][pm]
                for sc in range(prev_lo, prev_hi + 1):
                    ways = row[sc]
                    if ways:
                        # Try every current move cur != pm
                        for cur in (1, 2, 3):
                            if cur == pm:
                                continue
                            d = res[ai][cur]
                            nsc = sc + d
                            # We know nsc in [0..2n]
                            dp[nj][cur][nsc] = (dp[nj][cur][nsc] + ways) % MOD
                            if nsc < new_lo:
                                new_lo = nsc
                            if nsc > new_hi:
                                new_hi = nsc
            prev_lo, prev_hi = new_lo, new_hi

        # After n rounds, dp_layer holds dp[n%2]
        final_layer = dp[n & 1]
        # Sum sequences where Bob's score > Alice's => diff > 0 => offset > mid
        ans = 0
        # offsets valid are [prev_lo..prev_hi]
        for pm in (1, 2, 3):
            row = final_layer[pm]
            for sc in range(mid + 1, prev_hi + 1):
                ans = (ans + row[sc]) % MOD
        return ans