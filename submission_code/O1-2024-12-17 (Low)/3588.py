class Solution:
    def countWinningSequences(self, s: str) -> int:
        """
        We want the number of ways Bob can choose a creature each round (from {F,W,E}),
        never repeating the same creature consecutively, such that Bob's total points
        (according to the given matchup rules) exceed Alice's total points.

        Scoring rules (if players choose simultaneously A vs B):
          F vs E => F gets +1
          F vs W => W gets +1
          W vs E => E gets +1
        If both pick the same, score = 0.
        
        We translate those into Bob's perspective:
          scoreDiff(AliceCreature, BobCreature):
            (F, W) => +1  (Bob gains a point)
            (W, E) => +1
            (E, F) => +1
            (F, E) => -1  (Alice gains a point)
            (W, F) => -1
            (E, W) => -1
            same => 0

        We use a DP approach to count the number of valid Bob-sequences with final
        Bob-points minus Alice-points > 0.
        
        Let n = len(s).
        We map s[i] in { 'F','W','E' } to {0,1,2}, similarly Bob's moves.
        Let diff(a,b) be the round score from Bob's perspective.

        We define dp arrays of shape (2, 3, 2n+1).  (rolling over i and last-move)
        Index meaning:
          dp[cur][last][d] = how many ways Bob can reach round i with 'last' as Bob's
                             creature in round i-1, and Bob-Alice = d - offset,
                             where offset = n to keep array indices nonnegative.

        Transitions:
          For dp at round i -> dp at round i+1:
            If we are at dp[cur][last][d] and want to pick nextMove != last:
               new_d = d + diff( A[i], nextMove )

        Finally, sum all dp after n rounds for differences > 0.
        The time complexity is O(n * 3 * (2n) * 2) ~ O(6n^2), which is feasible for n=1000 
        with careful implementation in Python. 
        We take every count modulo 10^9+7.
        """

        MOD = 10**9 + 7
        n = len(s)

        # Map 'F','W','E' to 0,1,2
        mapping = {'F':0, 'W':1, 'E':2}
        alice = [mapping[ch] for ch in s]

        # scoreDiff[a][b] = difference for Bob if Alice=a, Bob=b
        #  F=0, W=1, E=2
        #  Using the rules:
        #   (F, W)=+1, (W, E)=+1, (E, F)=+1 
        #   (F, E)=-1, (W, F)=-1, (E, W)=-1 
        #   same=0
        scoreDiff = [[0]*3 for _ in range(3)]
        # Fill in the table explicitly:
        scoreDiff[0][0] = 0  # F vs F
        scoreDiff[0][1] = 1  # F vs W => W beats F => Bob +1
        scoreDiff[0][2] = -1 # F vs E => F beats E => Alice +1 => Bob -1

        scoreDiff[1][0] = -1 # W vs F => F beats W => Alice +1 => Bob -1
        scoreDiff[1][1] = 0  # W vs W
        scoreDiff[1][2] = 1  # W vs E => E beats W => Bob picks E => wait carefully:
                             # Actually W vs E => E beats W => if Alice=W, Bob=E => Bob +1
                             # But here "a=1, b=2" => "Alice=W, Bob=E" => E beats W => Bob +1 => correct: +1

        scoreDiff[2][0] = 1  # E vs F => F beats E => If Alice=E, Bob=F => Bob=F => F doesn't lose E? 
                             # Wait carefully:
                             # E vs F => If Alice=E, Bob=F => F vs E => F beats E => Bob=F => Bob gets +1 => correct is +1
        scoreDiff[2][1] = -1 # E vs W => E vs W => E beats W => if Alice=E, Bob=W => then W doesn't beat E => 
                             # Actually from Bob's perspective, Alice=E vs Bob=W => E beats W => Alice gets +1 => Bob -1 => correct
        scoreDiff[2][2] = 0  # E vs E

        offset = n  # so difference ranges from -n..n mapped to 0..2n
        sizeD = 2*n + 1

        # We'll use two layers of DP to save memory: dpCurr, dpNext, each shape(3, sizeD)
        dpCurr = [0]*(3*sizeD)
        dpNext = [0]*(3*sizeD)

        # Helper to index (creature, difference) in dp array:
        def idx(creature, diffIndex):
            return creature * sizeD + diffIndex

        # Initialize for i=0 (the first round).
        # Bob picks any of {0,1,2} with no previous constraint, difference = offset+scoreDiff(alice[0], bobMove)
        for bobFirst in range(3):
            d = offset + scoreDiff[alice[0]][bobFirst]
            dpCurr[idx(bobFirst,d)] += 1

        # Iterate from round i=1..n-1
        for i in range(1, n):
            # Clear dpNext
            for j in range(3*sizeD):
                dpNext[j] = 0

            a = alice[i]
            for lastCreature in range(3):
                baseIndex = lastCreature * sizeD
                for dVal in range(sizeD):
                    ways = dpCurr[baseIndex + dVal]
                    if ways == 0:
                        continue
                    # Try next creature for Bob, must differ from lastCreature
                    for nxt in range(3):
                        if nxt == lastCreature:
                            continue
                        newD = dVal + scoreDiff[a][nxt]
                        if 0 <= newD < sizeD:
                            dpNext[idx(nxt, newD)] = (dpNext[idx(nxt, newD)] + ways) % MOD

            # swap dpCurr, dpNext
            dpCurr, dpNext = dpNext, dpCurr

        # After n rounds, dpCurr has the counts for sequences of length n
        # We sum over lastCreature in 0..2 and differences d for which d-offset>0 => d>offset
        ans = 0
        for lastCreature in range(3):
            baseIndex = lastCreature * sizeD
            for dVal in range(offset+1, sizeD):
                ans = (ans + dpCurr[baseIndex + dVal]) % MOD

        return ans