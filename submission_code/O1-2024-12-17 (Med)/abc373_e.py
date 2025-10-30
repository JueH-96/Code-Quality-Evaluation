def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    A = list(map(int, input_data[3:]))

    # -------------------------------------------------------------------------
    # Quick edge case: if M == 0, that would mean no one can ever be elected;
    # but per the problem constraints 1 <= M <= N, so we skip that scenario.
    #
    # We have N candidates, total K votes, of which sumA = sum(A) have been counted.
    # There are R = K - sumA votes left to distribute.
    #
    # A candidate i (0-based in our code) with final votes Fi = A[i] + X
    # is guaranteed to be elected if, no matter how the remaining (R - X) votes
    # are distributed among the other (N-1) candidates, fewer than M candidates
    # can exceed Fi. Equivalently, it is impossible to pick M distinct j != i
    # whose final votes exceed Fi. In "worst case" terms, we check if we can
    # supply enough votes to M distinct others to surpass Fi; if that is
    # impossible, then i is guaranteed.
    #
    # As derived in the discussion, the standard method is:
    #   - Define "needed_j = max(0, (A[i] + X + 1) - A[j])" for j != i.
    #   - If we can choose M distinct j such that the sum of those needed_j
    #     is <= (R - X), then candidate i is NOT guaranteed (because we can
    #     push those M above i).  Otherwise, i is guaranteed.
    #
    # However, checking each i naively would be expensive.  A well-known shortcut:
    #   - Sort the candidates' vote counts in descending order (call this array B).
    #   - For each i, to see if M others can surpass (A[i] + X), we only need to
    #     consider the M "largest" A[j] (excluding i if i is among those top M).
    #     Because pushing the biggest opponents is "cheapest" in terms of needed votes
    #     to exceed i.
    #
    # Let sumA = sum(A).  Let R = K - sumA.
    #
    # For each i:
    #   1) If there are already M distinct j with A[j] > A[i] + R, then even if
    #      i uses all remaining R votes, i cannot catch or surpass them,
    #      so i can never be in the top M.  Answer = -1.
    #
    #   2) Otherwise, define T_i = the set of the M largest A[j] ignoring j=i.
    #      (If i is not in the top M+1 of the global sorted array, we just take
    #       the top M.  If i is inside the top M, we take the top (M+1) and exclude i.)
    #      We create a fast way to compute:
    #
    #         needed(i, X) = sum_{v in T_i} max(0, (A[i]+X+1) - v).
    #
    #      Then i is guaranteed iff needed(i, X) > (R - X).
    #
    #   3) We want the smallest X in [0..R] with needed(i,X) > R - X;
    #      or 0 if it already holds for X=0, or -1 if it does not hold even for X=R.
    #
    # Implementing this in O(N log N) plus an O(N log M log(R)) search can be
    # borderline large.  However, a common trick is:
    #   - We do a quick check for X=0. If needed(i,0) > R, then C_i=0.
    #   - We do a quick check for X=R. If needed(i,R) <= 0, then C_i=-1 (not possible).
    #   - Otherwise do a binary search over X in [0..R].  Checking needed(i,X) can be
    #     done in O(log M) via binary-search in T_i.  The total cost is roughly
    #     O(N * log M * log(R)).
    #
    # With N up to 2e5, M up to 2e5, and log(R) up to ~40 (because K up to 1e12),
    # this is indeed quite large in Python.  We will implement carefully and hope
    # that with efficient I/O and pruning, it can pass.  (In a lower-level language
    # with fast I/O, it is more comfortably done.)
    #
    # We'll do our best to optimize.  The main data structure steps:
    #
    #   - Sort all A in descending order with their indices.
    #   - Precompute prefix sums for the top M and for the top (M+1).
    #   - For each candidate i:
    #        * Check immediate impossibility (#1 above).
    #        * Determine the relevant set T_i (top M ignoring i or top M+1 ignoring i).
    #          Actually, we won't physically build T_i each time. We'll use
    #          logic with skip-index to handle it.
    #        * If needed(i,0) > R => answer=0
    #        * Else if needed(i,R) <= (R-R)=0 => answer=-1
    #        * Else do a binary search for minimal X.
    #
    # Let's implement.
    # -------------------------------------------------------------------------

    sys.setrecursionlimit(10**7)

    sumA = sum(A)
    R = K - sumA  # Remaining votes

    # Edge case: if M == N, then all candidates will be elected. Then everyone
    # is guaranteed already with X=0. We can just output 0 for all.
    # Because the condition: "the number of candidates who have received more votes
    # than i is less than M" => if M = N, then 'less than N' is always true unless
    # there are ties with i??? Actually with M=N, every candidate is elected
    # unconditionally. So C_i=0 for all i.
    if M == N:
        print(" ".join(["0"]*N))
        return

    # Build a list of (A[i], i) and sort descending by A[i].
    indexedA = [(A[i], i) for i in range(N)]
    indexedA.sort(key=lambda x: x[0], reverse=True)
    # B will be the sorted values in descending order (b_0 >= b_1 >= ...).
    # idxB will be the corresponding candidate indices in that order.
    B = [x[0] for x in indexedA]
    idxB = [x[1] for x in indexedA]

    # rank_of[i] = the 0-based rank of candidate i in the descending array B
    # i.e. if i appears at position p in idxB, rank_of[i] = p
    rank_of = [0]*N
    for r in range(N):
        rank_of[idxB[r]] = r

    # Precompute prefix sums for b[0..m-1] (top M) and also b[0..m] (top M+1) if M+1 <= N.
    # We'll store them in 0-based indexing, p0[k] = sum of the first k elements of b[0..k-1].
    # So p0[0] = 0, p0[1] = b[0], ...
    # We only need them if M>0 obviously; M>0 is guaranteed by the problem since M>=1.
    # Careful if M==N, we have handled that above, so here M<N. So top M and top M+1 exist.
    p0 = [0]*(M+1)  # prefix sums for the top M = b[:M]
    for i in range(M):
        p0[i+1] = p0[i] + B[i]

    # If M+1 <= N, we also build p1 for b[:M+1].
    # But if M == N-1, then M+1 = N, so we can build p1 safely if M+1 <= N (which it does).
    if M+1 <= N:
        p1 = [0]*(M+2)
        for i in range(M+1):
            p1[i+1] = p1[i] + B[i]
    else:
        p1 = None  # won't be needed if M+1 > N, but that can't happen unless M=N?

    # Function: can candidate i catch at least M others if i uses all R votes?
    # If there exist M distinct j with A[j] > A[i] + R, i can never surpass them.
    # Then answer is -1. We'll check this quickly:
    # Sort B descending => if B[M-1] > A[i]+R, that means the M-th largest is still > A[i]+R,
    # so there are at least M with > A[i]+R. Then i can't surpass them => -1
    # But we must also check if those top M are actually distinct from i.  If i is
    # in the top M, we check the M-th largest ignoring i or so.  Easiest way:
    #   - If rank_of[i] >= M => then the top M are b[0..M-1]. If b[M-1] > A[i]+R => -1
    #   - If rank_of[i] < M => then the M-th ignoring i is b[M] (the (M+1)-th largest).
    #     If b[M] > A[i] + R => -1
    # This handles quick impossibility.

    # We will build two helper functions to compute needed(i, X) for either set0 or set1.

    # set0 = b[0..M-1], prefix sums p0, size M
    # neededSet0(Ai, X) = sum of max(0, (Ai+X+1) - b[k]) for k in [0..M-1]

    # We'll do it by:
    #   1) T = Ai + X + 1
    #   2) find how many of b[0..M-1] >= T -> let that count be c
    #      (we do a binary search for T in b[0..M-1], descending.)
    #   3) needed = sum_{k=M-c..M-1} (T - b[k]) if b[k] < T, i.e. for those that are < T.
    #      We can compute sum of that subarray with prefix sums in O(1).

    import bisect

    def neededSet0(Ai, X):
        T = Ai + X + 1
        # We want the largest index where b[idx] >= T in b[0..M-1] (descending).
        # In Python's bisect, we have b sorted descending. We can invert T to do an ascending bisect,
        # or do a manual binary search. Let's implement quickly a custom binary search to find
        # count c of elements >= T.
        # c = number of elements b[k] >= T in b[:M].
        # Then the number of elements < T is M-c. We'll sum those < T.
        # We'll do a simple approach: we'll search for T in descending array.
        lo, hi = 0, M  # searching in b[:M]
        while lo < hi:
            mid = (lo+hi)//2
            if B[mid] >= T:
                lo = mid+1
            else:
                hi = mid
        c = lo  # c = # of elements >= T
        # Sum of those < T is the subarray b[c..M-1]. Let that subarray have length L = M-c.
        # needed = (M-c)*T - sum(b[c..M-1])
        sum_b_sub = p0[M] - p0[c]  # sum of b[c..M-1]
        L = M - c
        return L*T - sum_b_sub

    # set1(r) = b[0..M] ignoring b[r], so effectively we have M elements left.
    # We'll define neededSet1(r, Ai, X).  The array is of length M+1 in descending order b[0..M].
    # We skip the index r (0-based within that subarray).  Then do the same logic:
    #   1) T = Ai+X+1
    #   2) find how many among those M elements are >= T
    # Implementation detail: we do a binary search in b[0..M], but we skip b[r].
    # We'll do it carefully:
    #
    #   - First find how many among b[0..M] (size M+1) are >= T in a normal sense.
    #     Let that be c'.  Then we see if r is in the range of those c'.  If so, we reduce c' by 1
    #     because we skip that index.  That yields c = the number of elements >= T ignoring b[r].
    #
    #   - Then the subarray of < T ignoring b[r] has length M - c.  We'll find its sum.  We can do:
    #       sum0 = (sum of b[0..M]) - b[r].
    #       Then sum of the top c ignoring b[r] is sum of the top c' from b[0..M] minus b[r] if
    #       b[r] is in that top c' subset.  It's simpler to think in terms of the combined approach:
    #
    # We'll do it step by step to avoid confusion.

    # Precompute sum_of_b0_Mplus1 = p1[M+1] = sum of b[0..M] if p1 is not None.
    if p1 is not None:
        sum_of_b0_Mplus1 = p1[M+1]
    else:
        sum_of_b0_Mplus1 = 0  # Not used if M+1> N anyway.

    def neededSet1(r, Ai, X):
        # r in [0..M], we skip b[r].  So the set we consider has M elements total.
        # T = Ai + X + 1
        T = Ai + X + 1
        # first, find c' = number of b[k] >= T for k in [0..M].
        # We'll do a custom binary search in b[:M+1].
        lo, hi = 0, M+1
        while lo < hi:
            mid = (lo+hi)//2
            if B[mid] >= T:
                lo = mid+1
            else:
                hi = mid
        cprime = lo  # # of elements >= T in b[:M+1]
        # Now skip index r.  If r < cprime, then b[r] was counted among those >= T,
        # so c = cprime - 1.  Else c = cprime.
        if r < cprime:
            c = cprime - 1
        else:
            c = cprime
        # Among the M elements ignoring b[r], exactly c have value >= T.
        # So the number with value < T is (M - c).
        # Next, sum of those < T ignoring b[r].
        # sum0 = sum(b[0..M]) - b[r].
        sum0 = sum_of_b0_Mplus1 - B[r]
        # The sum of the c largest ignoring b[r] can be found if we had a prefix sums
        # with skip.  But let's do a small trick:
        # The c largest ignoring b[r] is basically c largest from b[0..M], skipping r.
        #
        # Let's define: sum_of_top_cprime_in_b0Mplus1 = p1[cprime].
        # If r < cprime, then we skip b[r] from that top cprime sum, hence sum_of_top_c = p1[cprime] - b[r].
        # Otherwise sum_of_top_c = p1[cprime].
        #
        # Then c = cprime or cprime-1. Actually, we just want sum_of those c ignoring b[r].
        # Let sum_top_cprime = p1[cprime].
        sum_top_cprime = p1[cprime]

        if r < cprime:
            sum_top_c = sum_top_cprime - B[r]
        else:
            sum_top_c = sum_top_cprime

        # But we must be sure that c = cprime or cprime-1. Actually let's check:
        # if r < cprime, c = cprime-1 => we want the top (cprime-1) ignoring r => that sum is sum_top_cprime - B[r] minus possibly the next largest element?
        # Wait, we have to be consistent.  Actually, c = number of elements >= T ignoring r.  But that doesn't necessarily match cprime-1 exactly for top elements, because we are pivoting at T. This is tricky if T is exactly equal to some b[r].
        #
        # A simpler approach: we don't actually need the sum of the top c. We want the sum of those < T ignoring r. That is sum0 minus the sum of those >= T ignoring r. Let's define it that way:
        #
        # The sum of those < T ignoring r] = sum0 - sum_of_those_>=T_ignoring_r.
        # We need sum_of_those_>=T_ignoring_r.  That is the sum of c elements from the set b[0..M] \ {b[r]} that are >= T. Let's gather them from the top c' or c'+1 approach. We'll do:
        #
        #   sum_of_those_>=T_in_b0Mplus1 = sum_top_cprime
        #   if r < cprime, then we exclude b[r], so sum_of_those_>=T_ignoring_r = sum_top_cprime - b[r].
        # else sum_of_those_>=T_ignoring_r = sum_top_cprime.
        #
        # but c might be cprime or cprime-1. Indeed, if r < cprime, we lose 1 from cprime => c = cprime-1. So the sum of those >= T ignoring r has size c. We expect c = cprime or cprime-1. So if r< cprime => c = cprime-1 => the sum is sum_top_cprime - b[r]. That has exactly (cprime-1) elements. Good.
        # if r >= cprime => c = cprime => the sum is sum_top_cprime. That has cprime elements. Good.
        #
        # So define:
        if r < cprime:
            sum_ge = sum_top_cprime - B[r]
        else:
            sum_ge = sum_top_cprime

        # Then sum of < T ignoring r = sum0 - sum_ge
        sum_lt = sum0 - sum_ge
        # number < T ignoring r = M - c
        L = M - c
        # needed = L*T - sum_lt
        needed_val = L*T - sum_lt
        return needed_val

    # We'll implement a function get_needed(i, X) that returns needed(i, X) using the sets logic
    # and skipping i if it's in the top M (or M+1).
    def get_needed(i, X):
        Ai = A[i]
        r = rank_of[i]  # 0-based rank in B
        # Are we using set0 or set1?
        # if r >= M: use set0 => b[:M]
        # else use set1(r) => b[:M+1], skipping r
        if r >= M:
            # set0
            return neededSet0(Ai, X)
        else:
            # set1(r)
            # but we have to be careful which r we pass. The subarray b[:M+1] includes b[r] as the r-th
            # in 0-based indexing. So we pass r as is.
            return neededSet1(r, Ai, X)

    # Also define a quick check for the "already M strictly bigger than A[i]+R" => immediate -1
    # if r >= M, the M-th largest ignoring i is b[M], else it is b[M] if r< M? Actually see logic:
    # If r >= M, the top M are b[0..M-1]. The M-th largest is b[M-1], but 0-based => that's b[M-1].
    # We want to see if that is strictly > A[i]+R. Actually we want to see if there are M distinct j
    # with A[j] > A[i]+R.
    #
    # Let us define a small helper: top M distinct j ignoring i means:
    #   if r < M => that set's M-th largest is b[M], because we skip i in the top M+1 set,
    #   otherwise => the M-th largest is b[M-1].
    #
    # But we don't just check the M-th largest; they could all be bigger. We need to check if the candidate
    # at position M-1 or M can still be > A[i]+R. We'll do it precisely:
    #
    # The condition "there exist M j with A[j] > A[i]+R" is the same as
    # "the M-th largest A[j] ignoring i is > A[i]+R".
    #
    # If r < M:
    #   then ignoring i among the top M+1 => the M-th largest ignoring i is b[M], because we skip b[r].
    #   if b[M] > A[i]+R => -1
    # else
    #   ignoring i among b[0..M-1], the M-th largest is b[M-1], if b[M-1] > A[i]+R => -1.
    #
    # We must also check if M-1 or M is in range. If M-1 >= N, no, but that can't happen because M<=N-1
    # (since M<N if the code got here).
    if M <= N-1:
        # We'll define a function to check this quickly
        def cannot_win_if_topM_too_big(i):
            r = rank_of[i]
            Ai = A[i]
            if r < M:
                # check b[M] if M < N
                if M < N and B[M] > Ai + R:
                    return True
                else:
                    return False
            else:
                # check b[M-1]
                if B[M-1] > Ai + R:
                    return True
                else:
                    return False
        # We'll use this in the main loop below.
    else:
        # M == N, handled above, so we won't get here.
        def cannot_win_if_topM_too_big(i):
            return False

    # We'll now produce the answer array
    ans = [-1]*N

    # We will implement the solve_for_candidate(i) that sets ans[i].
    # Steps:
    # 1) If cannot_win_if_topM_too_big(i) => -1
    # 2) Else check if needed(i,0) > R => ans[i]=0
    # 3) Else check if needed(i,R) <= 0 => ans[i]=-1
    # 4) Else do a binary search in [0..R] for the smallest X s.t. needed(i,X) > R - X.

    def solve_for_candidate(i):
        # Step 1
        if cannot_win_if_topM_too_big(i):
            return -1

        # Step 2: check if needed(i,0) > R
        n0 = get_needed(i, 0)
        if n0 > R:
            return 0

        # Step 3: check if needed(i,R) > R - R = 0
        nR = get_needed(i, R)
        if nR <= 0:
            # can't guarantee
            return -1

        # Step 4: binary search for minimal X
        # We want needed(i,X) > R - X
        # define f(X) = needed(i,X) - (R - X) = needed(i,X) + X - R
        # We want f(X) > 0
        # We'll do standard integer binary search in [0..R].
        low, high = 0, R
        while low < high:
            mid = (low+high)//2
            val = get_needed(i, mid)
            if val > (R - mid):
                high = mid
            else:
                low = mid + 1
        if low <= R:
            return low
        else:
            return -1

    # We'll compute answers:
    # This will cost O(N log(R) log(M)) in worst case.  We rely on some pruning
    # to handle large test data.  We'll do our best to optimize.

    # Prepare a fast output buffer:
    import math

    out = []
    for i in range(N):
        c_i = solve_for_candidate(i)
        out.append(str(c_i))

    print(" ".join(out))