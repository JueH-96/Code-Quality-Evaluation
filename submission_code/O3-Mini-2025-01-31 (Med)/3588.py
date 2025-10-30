MOD = 10**9 + 7

class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        # Map creature char to index: F->0, W->1, E->2.
        char_to_index = {'F': 0, 'W': 1, 'E': 2}
        # Outcome matrix: outcome[b][a] stores the score change for Bob (i.e. Bob_point - Alice_point)
        # when Bob uses creature b and Alice uses creature a.
        # Rules: 
        #   • Fire Dragon (F) beats Earth Golem (E)
        #   • Water Serpent (W) beats Fire Dragon (F)
        #   • Earth Golem (E) beats Water Serpent (W)
        # Ties give 0.
        #
        # Mapping our indices F=0, W=1, E=2:
        # Bob F:   vs F: tie (0), vs W: lose (-1), vs E: win (+1)
        # Bob W:   vs F: win (+1), vs W: tie (0), vs E: lose (-1)
        # Bob E:   vs F: lose (-1), vs W: win (+1), vs E: tie (0)
        outcome = [
            [0, -1, 1],  # Bob chooses F
            [1,  0, -1], # Bob chooses W
            [-1, 1,  0]  # Bob chooses E
        ]
        
        # We use dynamic programming where the state is:
        #   (last_move, diff)
        # where diff = (Bob's points) - (Alice's points)
        # Bob is not allowed to pick the same move consecutively.
        # Since total rounds n, the difference diff can vary between -n and n.
        # We use an offset to index differences in our DP table.
        offset = n
        size = 2 * n + 1   # indices 0 .. 2*n correspond to diff values -n ... n.
        
        # dp[move][d] holds the number of ways to obtain a state ending with Bob move "move" 
        # and having a total difference = d - offset after a given round.
        dp = [[0] * size for _ in range(3)]
        
        # Process round 0 separately (Bob can pick any creature)
        a0 = char_to_index[s[0]]
        for bob in range(3):
            diff = outcome[bob][a0]
            dp[bob][diff + offset] = (dp[bob][diff + offset] + 1) % MOD
        
        # Process rounds 1 ... n-1.
        for i in range(1, n):
            a = char_to_index[s[i]]
            newdp = [[0] * size for _ in range(3)]
            for prev_move in range(3):
                for d in range(size):
                    count = dp[prev_move][d]
                    if count:
                        # Bob's next move must be different from prev_move.
                        for next_move in range(3):
                            if next_move == prev_move:
                                continue
                            delta = outcome[next_move][a]
                            nd = d + delta  # new index = old difference index + delta
                            if 0 <= nd < size:
                                newdp[next_move][nd] = (newdp[next_move][nd] + count) % MOD
            dp = newdp
        
        # Bob beats Alice if the total number of points for Bob is strictly greater than for Alice 
        # i.e. if (Bob_points - Alice_points) > 0.
        # Since diff = d - offset, we need d > offset.
        ans = 0
        for move in range(3):
            for idx in range(offset + 1, size):
                ans = (ans + dp[move][idx]) % MOD
        return ans


# If you want to run some tests, you can use the following:
if __name__ == "__main__":
    sol = Solution()
    # Example tests:
    tests = [
        ("FFF", 3),
        ("FWEFW", 18)
    ]
    for s, expected in tests:
        result = sol.countWinningSequences(s)
        print(f"s = {s}, result = {result}, expected = {expected}")