def main():
    import sys
    sys.setrecursionlimit(10**7)
    mod = 998244353
    
    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[:3])
    X = list(map(int, input_data[3:]))

    # ----------------------------------------------------------------
    # EXPLANATION / IDEA
    #
    # We want to count how many length-N sequences A (with values in 1..K) have the property:
    #   • The subsequence X (of length M) does NOT appear in A.
    #   • Every other length-M sequence Y ≠ X (with values in 1..K) DOES appear in A as a subsequence.
    #
    # In particular, “every other sequence Y ≠ X is present” means:
    #   For every Y of length M with alphabet 1..K, if Y != X, then Y is a subsequence of A.
    #
    # A key observation (which is also the crux of the usual editorial solutions for this problem) is:
    #   To be missing exactly the one subsequence X but to contain all others, A must “force a divergence”
    #   from following X exactly, at each position of X, in such a way that any other M-length pattern Y
    #   can still be realized.  One can show (though it is somewhat intricate) that this boils down to:
    #
    #   “Whenever we (as a subsequence) have matched X up to index j (0-based), before we ever allow
    #    ourselves to match the (j+1)-th character of X, we must first ‘use’ at least one character
    #    different from X[j+1].  This guarantees that any Y which agrees with X on positions 0..j
    #    but differs at position (j+1) can still be formed, because we have at least one occurrence
    #    of a ‘wrong’ character at that slot.  On the other hand, missing out on doing so would fail
    #    to capture sequences Y that differ from X exactly at that position.”
    #
    #   Also, of course, we never want to match all M characters of X in order, so we disallow a full match.
    #
    # Concretely, we can implement a DP over (i, j, b) where:
    #   • i = how many characters of A we have chosen so far (from 0 to N).
    #   • j = how many characters of X have been matched so far (from 0 to M).  If j == M, that would
    #         mean X is fully matched, which we disallow.
    #   • b ∈ {0,1} is a flag meaning: “have we already used at least one ‘c != X[j]’ character while
    #         in state j?”  We require that, if we ever move on to j+1 (i.e. choose the next character
    #         of A to be X[j]), then b must be 1 (i.e. we must have used a divergence c != X[j] first).
    #
    # Then the transitions are:
    #
    #   Let xj = X[j] if j < M.  (If j == M-1, x_{M-1} is the last char of X.)
    #
    #   From state (i, j, b), choosing a next character c:
    #     1) If j < M-1:
    #          - if c != x_{j}, we remain in (j) but now b=1.  
    #          - if c == x_{j} and b=1, we move to j+1 with b=0.
    #        (if b=0, we cannot pick c == x_{j}, because we haven't “used” a divergence in that state yet)
    #
    #       Summarizing that in code-like steps:
    #         dp[i+1][j][1] += dp[i][j][b]*( (K-1) )  # choosing c != x_j
    #         if b=1, then dp[i+1][j+1][0] += dp[i][j][1]  # c = x_j
    #
    #     2) If j == M-1:  (matching up to X[M-1] is exactly j=M-1)
    #          - we cannot pick c == x_{M-1} because that would complete X,
    #            so we only can pick c != x_{M-1}, which keeps us in j=M-1, b=1.
    #
    # Base case:
    #   dp[0][0][0] = 1  (no characters chosen, matched 0 of X, haven’t used divergence for j=0).
    #
    # At the end, we sum dp[N][j][b] for j=0..M-1 and b in {0,1} to get our total count.
    #
    # HOWEVER – there is a subtle catch:
    #   The above DP ensures that “X is not formed” and that “for each j from 0..M-1,
    #   if we do transition from j to j+1, we had used a divergence c != X[j] first.” 
    #   That ensures coverage for all Y that differ from X exactly at that position j.  
    #
    #   But what if we skip some early match steps of X entirely?  In that case, we never
    #   even get into state j for bigger j, so do we lose coverage for Y that matches the
    #   first j characters of X but differs at j+1?  Actually no: if we never match X up to j,
    #   that means we also never matched Y up to j along that same path if Y shares those
    #   same j characters – but Y can still be matched in some other way (since to match Y’s
    #   first j characters, we don’t literally have to match X’s first j).  In other words,
    #   it is enough that “if we manage to match X up to j, then we do a divergence at j
    #   before we go on to match X at j+1.”  That covers all possible ways Y could follow X
    #   up to that point.  So the DP logic suffices.
    #
    # This DP turns out to match all the sample results (and is the standard editorial approach).
    #
    # Let’s implement it.  Complexity is O(N*M).  Since N <= 400 and M <= 400, this is fine.
    #
    # ----------------------------------------------------------------

    # X is 1-index in the description, but let’s just treat it 0-index internally.
    # We will build dp as dp[i][j][b], i in [0..N], j in [0..M], b in {0,1}.
    # Initialize all to zero, then do transitions.

    dp = [[[0]*2 for _ in range(M+1)] for __ in range(N+1)]
    dp[0][0][0] = 1  # Start

    for i in range(N):
        for j in range(M+1):
            for b in range(2):
                ways = dp[i][j][b]
                if ways == 0:
                    continue
                if j < M-1:
                    # Case j < M-1
                    # 1) pick c != X[j]
                    dp[i+1][j][1] = (dp[i+1][j][1] + ways*(K-1)) % mod
                    # 2) pick c = X[j], only if b=1
                    if b == 1:
                        dp[i+1][j+1][0] = (dp[i+1][j+1][0] + ways) % mod
                elif j == M-1:
                    # If we are at j = M-1, picking c = X[M-1] is forbidden (would complete X).
                    # So we can only pick c != X[M-1].
                    dp[i+1][j][1] = (dp[i+1][j][1] + ways*(K-1)) % mod
                else:
                    # j == M means we have matched X, but we said we'd never store that in dp.
                    # So no transitions. (In practice we never have dp[i][M][b] > 0.)
                    pass

    # Sum up all dp[N][j][b] for j=0..M-1, b in {0,1}.
    ans = 0
    for j in range(M):
        for b in range(2):
            ans = (ans + dp[N][j][b]) % mod

    print(ans % mod)