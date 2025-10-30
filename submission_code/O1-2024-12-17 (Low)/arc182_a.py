def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data=sys.stdin.read().strip().split()
    N=int(input_data[0])
    Q=int(input_data[1])
    P=[0]*Q
    V=[0]*Q
    idx=2
    for i in range(Q):
        P[i]=int(input_data[idx]); V[i]=int(input_data[idx+1])
        idx+=2

    MOD = 998244353

    # ----------------------------------------------------------------
    # Explanation of the approach (high-level):
    #
    # We have Q operations in chronological order. Each operation i
    # must be chosen as "prefix" (overwrite S[1..P_i]) or "suffix"
    # (overwrite S[P_i..N]). The "no crying" constraints mean:
    #
    # 1) If we pick a prefix at step i, then in that overwritten region
    #    (1..P_i) there must be no element strictly greater than V_i.
    #    Equivalently, there cannot exist a previous operation k < i
    #    with V_k > V_i that overwrote any index j in [1..P_i].
    #    - A prefix k overwrote [1..P_k].
    #      That region definitely includes j=1 (since P_k ≥ 1).
    #      So if V_k > V_i and operation k is prefix, it always
    #      intersects [1..P_i].  → So having even one earlier prefix k
    #      with V_k>V_i makes prefix i impossible.
    #    - A suffix k overwrote [P_k..N].
    #      This intersects [1..P_i] if and only if P_k ≤ P_i.
    #      So if V_k > V_i and operation k is suffix with P_k ≤ P_i,
    #      that blocks prefix i.
    #
    # 2) If we pick a suffix at step i, then in that overwritten region
    #    (P_i..N) there must be no element strictly greater than V_i.
    #    Equivalently, there cannot exist a previous operation k < i
    #    with V_k > V_i that overwrote any index j in [P_i..N].
    #    - A suffix k overwrote [P_k..N]. That always intersects
    #      [P_i..N] for any P_k ≤ N (always some overlap).
    #      → So having even one earlier suffix k with V_k>V_i
    #         makes suffix i impossible.
    #    - A prefix k overwrote [1..P_k].
    #      This intersects [P_i..N] if P_k ≥ P_i.
    #      So if V_k > V_i and operation k is prefix with P_k ≥ P_i,
    #      that blocks suffix i.
    #
    # From these rules, if we track only which earlier prefix-ops
    # have V_k>current and the maximum P_k among them (call it pmax),
    # and which earlier suffix-ops have V_k>current and the minimum
    # P_k among them (call it smin), we can express feasibility of
    # choosing prefix/suffix at step i succinctly:
    #
    # Let "prefix-blockers" be earlier prefix ops with value bigger
    # than the current V_i (their indices in the chronological sense).
    # All of them block prefix i. Hence if prefix-blockers is nonempty,
    # we cannot do prefix i at all.
    #
    # Let "suffix-blockers" be earlier suffix ops with value bigger
    # than current V_i. All of them block suffix i. Hence if
    # suffix-blockers is nonempty, we cannot do suffix i at all.
    #
    # Additionally, we must ensure intersection is avoided for the
    # other type of operation:
    #   - If we choose prefix i, we must ensure that all suffix-blockers
    #     have P_k > P_i (so that [P_k..N] does not intersect [1..P_i]).
    #     Equivalently, the minimum P_k among suffix-blockers is smin.
    #     We need smin > P_i.
    #
    #   - If we choose suffix i, we must ensure that all prefix-blockers
    #     have P_k < P_i (so that [1..P_k] does not intersect [P_i..N]).
    #     If pmax is the maximum P_k among prefix-blockers, we need
    #     pmax < P_i.
    #
    # Then, once we choose prefix i (with V_i), if V_i is strictly
    # bigger than some future operation's V_j, i becomes a "prefix-blocker"
    # for that future j. Similarly if we choose suffix i and V_i is
    # strictly bigger than some future operation's V_j, i becomes a
    # "suffix-blocker" for that j. But we do not want to pass blockers
    # forward "per future j" individually; we want an aggregated state.
    #
    # A known technique is to build a DP over i plus two integers (pmax, smin)
    # that indicate how large the prefix-blockers' P_k can be (pmax = max(P_k))
    # and how small the suffix-blockers' P_k can be (smin = min(P_k)).
    # pmax=0 means we have no prefix-blockers so far, smin=N+1 means
    # we have no suffix-blockers so far. We must maintain pmax < smin
    # always (otherwise it is an invalid state).
    #
    # Let block[i] be a boolean indicating whether operation i's value
    # is strictly larger than at least one future operation's value
    # (i.e. does V_i > min(V_j for j>i)?). If not, then picking i as
    # prefix or suffix won't block anything. If yes, then picking i
    # as prefix sets pmax = max(pmax, P_i), picking i as suffix sets
    # smin = min(smin, P_i).
    #
    # Transitions (processing i from 1 to Q in order):
    #
    #   Suppose our state before step i is (pmax, smin).
    #   Let p = P_i, b = block[i]. We can pick:
    #
    #   1) prefix if and only if pmax=0 (no prefix-blockers yet)
    #      and smin > p (suffix-blockers' minimum P_k is > p).
    #      Then the next state is:
    #         new_pmax = b ? p : pmax  (which is b ? p : 0, since pmax=0)
    #         new_smin = smin (unchanged)
    #      We add dp[i, pmax, smin] to dp[i+1, new_pmax, new_smin].
    #
    #   2) suffix if and only if smin = N+1 (no suffix-blockers yet)
    #      and pmax < p (prefix-blockers' maximum P_k < p).
    #      Then the next state is:
    #         new_pmax = pmax (unchanged)
    #         new_smin = b ? p : smin (which is b ? p : N+1, since smin=N+1)
    #      We add dp[i, pmax, smin] to dp[i+1, new_pmax, new_smin].
    #
    # We'll denote dp[i][pmax][smin] as the number of ways to pick
    # operations from step 1..i under those blocker conditions at the
    # end of step i. Our answer is the sum of dp[Q][pmax][smin] over
    # all valid pmax < smin.
    #
    # Implementing this 3D DP directly is large (Q*N*N up to 25e7 states),
    # which is borderline but might be done carefully in C++ with
    # optimizations. In Python, we must do it more cleverly with
    # prefix/suffix sums to aggregate transitions in O(N) per i
    # rather than O(N*N).
    #
    # We will store dp[i] in a 2D array dp[i][pmax][smin], but that
    # is too big for Python memory. Instead we keep two 2D layers:
    # dpCur[pmax][smin] for the end of step i, and dpNxt[pmax][smin]
    # for the end of step i+1. Then we do transitions. We rely on
    # prefix sums along the pmax or smin dimension.
    #
    # Specifically:
    #  - For choosing prefix at step i, we require pmax=0 and smin>p.
    #    Then dpNxt[new_pmax][smin] += sum of dpCur[0][x] for x>p.
    #    new_pmax = (b? p : 0).
    #
    #  - For choosing suffix at step i, we require smin=N+1 and pmax<p.
    #    Then dpNxt[pmax][new_smin] += sum of dpCur[u][N+1] for u<p.
    #    new_smin = (b? p : N+1).
    #
    # We'll implement two prefix-sum arrays over dpCur:
    #   sumSminAbove[pmax][x] = sum of dpCur[pmax][s] for s > x
    #   sumPmaxBelow[x][smin] = sum of dpCur[u][smin] for u < x
    #
    # Then we can do each transition in O(1), and building these sums
    # in O(N) for each pmax or smin. Overall O(Q * N) or O(Q*N) with care.
    # This should be around 25e6 operations, which is on the edge but
    # might be done in optimized Python/PyPy. We will attempt to be
    # efficient.
    #
    # Implementation details below.
    # ----------------------------------------------------------------

    # Step 1: Precompute block[i] = (V[i] > min_{j>i} V[j])?
    # We'll store them in 0-based indexing for convenience in code.
    from math import inf

    min_from_right = [10**18]*(Q+2)
    min_from_right[Q] = 10**18
    min_from_right[Q+1] = 10**18
    for i in range(Q-1, -1, -1):
        min_from_right[i] = min(min_from_right[i+1], V[i])

    block = [False]*Q
    for i in range(Q):
        if i == Q-1:
            # last operation has no future
            block[i] = False
        else:
            block[i] = (V[i] > min_from_right[i+1])

    # dpCur[pmax][smin], pmax in [0..N], smin in [0..N+1], pmax < smin
    # We'll store in a 2D array of size (N+1) x (N+2).
    # Then we build dpNxt similarly. Initialize dpCur for i=0 as
    # dpCur[0][N+1] = 1  (no operations chosen yet, no blockers).
    # Then proceed i=1..Q.

    dpCur = [[0]*(N+2) for _ in range(N+1)]
    dpCur[0][N+1] = 1  # before any operations, one way, with no blockers

    for i in range(Q):
        # We'll compute dpNxt from dpCur using the transitions for operation i
        dpNxt = [[0]*(N+2) for _ in range(N+1)]
        p = P[i]
        b = block[i]

        # Build partial sums for dpCur, to handle "prefix" choice:
        #   We want sum of dpCur[0][s] for s > p.
        # Let's do a running sum from right to left for row pmax=0.
        rowSum = 0
        smin_above_p = [0]*(N+2)  # smin_above_p[x] = sum of dpCur[0][s] for s> x
        currow = dpCur[0]
        running = 0
        for s in range(N+1, -1, -1):
            running = (running + currow[s])%MOD
            smin_above_p[s] = running

        # Build partial sums for dpCur, to handle "suffix" choice:
        #   We want sum of dpCur[u][N+1] for u < p.
        colSum = 0
        pmax_below_p = [0]*(N+1)  # pmax_below_p[x] = sum of dpCur[u][N+1] for u< x
        running = 0
        for u in range(p):
            running = (running + dpCur[u][N+1])%MOD
            pmax_below_p[u+1] = running
        for u in range(p, N+1):
            pmax_below_p[u] = pmax_below_p[p]

        # Now do transitions:

        # 1) prefix choice:
        #    requires pmax=0 and smin>p
        #    dpNxt[new_pmax][smin] += sum_{s>p} dpCur[0][s]
        #    new_pmax = (b? p : 0)
        if True:
            # sum_{s>p} dpCur[0][s] = smin_above_p[p+1]
            ways_prefix = smin_above_p[p+1]  # all s>p
            if ways_prefix:
                new_pmax = p if b else 0
                # We add ways_prefix to dpNxt[new_pmax][s] for all s>p?
                # Actually the smin stays the same as in the old states?  No,
                # each old state has a distinct s>p, we do not change smin
                # with a prefix operation. So for every s>p that had dpCur[0][s],
                # we add that same count to dpNxt[new_pmax][s]. But we want
                # a single aggregate. So let's just do:
                # dpNxt[new_pmax][s] += dpCur[0][s] for s>p.
                # We can do it by a partial-sums difference as well.
                # We'll do it explicitly in a second pass:

                # Let's do one pass: for s in p+1..N+1 we do:
                # dpNxt[new_pmax][s] += dpCur[0][s].
                currow = dpCur[0]
                outrow = dpNxt[new_pmax]
                for s in range(p+1, N+2):
                    if currow[s]:
                        outrow[s] = (outrow[s] + currow[s])%MOD

        # 2) suffix choice:
        #    requires smin=N+1 and pmax<p
        #    dpNxt[pmax][new_smin] += sum_{pmax<p} dpCur[pmax][N+1].
        #    new_smin = (b? p : N+1).
        if True:
            # For each pmax in [0..p-1], we add dpCur[pmax][N+1].
            # The sum of dpCur[u][N+1] for u<p is pmax_below_p[p].
            # But each pmax in [0..p-1] transitions to the same new_smin = (b?p:N+1).
            # So dpNxt[pmax][ new_smin ] += dpCur[pmax][N+1].
            # We'll do it in a second pass, but we accumulate partial sums for speed.

            new_smin_val = p if b else (N+1)

            # We'll go pmax=0..p-1, dpNxt[pmax][new_smin_val] += dpCur[pmax][N+1]
            # We'll need dpCur[pmax][N+1] individually though. Let's just do a loop:
            outcol = [0]*(p)  # store partial sums for dpCur[x][N+1] x=0..p-1
            for x in range(p):
                outcol[x] = dpCur[x][N+1]

            # Now we add them to dpNxt[x][new_smin_val].
            for x in range(p):
                val = outcol[x]
                if val:
                    dpNxt[x][new_smin_val] = (dpNxt[x][new_smin_val] + val)%MOD

        # done computing dpNxt => move dpNxt to dpCur
        dpCur = dpNxt

    # At the end, sum dpCur[pmax][smin] for all pmax < smin.
    ans = 0
    for pmax in range(N+1):
        for smin in range(pmax+1, N+2):
            ans = (ans + dpCur[pmax][smin])%MOD

    print(ans % MOD)