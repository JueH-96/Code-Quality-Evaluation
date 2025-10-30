def main():
    import sys,sys
    data = sys.stdin.buffer.read().strip().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    total = sum(A)
    R = K - total  # remaining votes
    # Special case: if M == N, then every candidate is elected.
    # (Because condition “# of candidates with more votes < N” always holds.)
    if M == N:
        sys.stdout.write(" ".join("0" for _ in range(N)))
        return

    # We'll need a global sorted array B of all candidates’ votes, along with each candidate's original index.
    arr = [(A[i], i) for i in range(N)]
    arr.sort(key=lambda x: x[0])
    # Global sorted votes (non-decreasing order)
    global_B = [x[0] for x in arr]
    # pos_in_B[i] will be candidate i’s index (position) in global_B.
    pos_in_B = [0]*N
    for pos, (_, i) in enumerate(arr):
        pos_in_B[i] = pos

    # Build prefix sum array for global_B.
    prefix = [0]*(N+1)
    for i in range(N):
        prefix[i+1] = prefix[i] + global_B[i]

    # We'll use bisect_right from python's bisect module.
    from bisect import bisect_right

    # Explanation of safety check (for candidate i with extra votes X):
    # Let candidate i’s final votes be T = A[i] + X.
    # Then let pos = bisect_right(global_B, T). (global_B is the sorted list of all candidates’ current votes.)
    # Since candidate i always has A[i] <= T (because X >= 0) candidate i is definitely in global_B[0:pos].
    # Thus the number of opponents (other candidates) with A[j] ≤ T is (pos – 1)
    # and those with A[j] > T give free boosts (free_count = N – pos).
    # To “lose” candidate i, an adversary must boost r = M – free_count opponents among the ones with A[j] ≤ T.
    # The minimum total boost cost for r opponents is
    #     r*(T+1) – S_eff,
    # with S_eff defined as the sum of the r highest votes among the eligible opponents.
    # (Since when an opponent j is “eligible” (i.e. A[j] ≤ T) its cost is d_j = T – A[j] + 1,
    #  and the cheapest r are those with highest A[j].)
    # Candidate i is safe if:
    #             r*(T+1) – S_eff > (R – X)
    # which (after some algebra) is equivalent to:
    #             (r+1)*T > R + S_eff + A[i] – r.
    #
    # Note: in the computation of S_eff (the sum of the r highest eligible opponents’ votes)
    # we use our global sorted list. In global_B the “r highest” in indices [0,pos]
    # would be the last r numbers in that block. (Recall that candidate i is among global_B[0:pos],
    # so when “removing” candidate i, we have to adjust if candidate i lies in the block of r numbers.)
    #
    # We now binary search on X (0 ≤ X ≤ R) for the minimal extra votes making candidate i safe.
    
    res = [None]*N
    # To speed up inner loops we “bind” some variables.
    globB = global_B
    pref = prefix
    n = N
    bisect_r = bisect_right
    pos_in = pos_in_B

    # For each candidate:
    for i in range(N):
        base = A[i]
        idx = pos_in[i]  # candidate i's index in globB
        # Binary search over X in [0, R] 
        lo = 0
        hi = R + 1
        while lo < hi:
            mid = (lo + hi) >> 1
            T = base + mid
            pos = bisect_r(globB, T)
            # free_count = number of opponents with vote > T = n - pos.
            # Candidate i is safe only if free_count < M.
            if pos <= n - M:
                lo = mid + 1
                continue
            # Let r = M - free_count = pos + M - n.
            r = pos + M - n
            # The block global_B[0:pos] contains all candidates with A[j] <= T.
            # But candidate i is included in that block. The r highest votes from the eligible opponents
            # (i.e. excluding candidate i) are what the adversary would “buy”.
            # Compute the sum S_eff of the r largest eligible opponents.
            L = pos - r  # The r largest numbers in global_B[0:pos] are normally globB[L:pos]
            if idx >= L:
                # Candidate i is in this block – remove its vote and “replace” it with the next candidate.
                S_eff = (pref[pos] - pref[L]) - globB[idx] + globB[L - 1]
            else:
                S_eff = pref[pos] - pref[L]
            # Now check the safety condition:
            # (r+1)*T > R + S_eff + base - r .
            if (r + 1) * T > R + S_eff + base - r:
                hi = mid
            else:
                lo = mid + 1
        if lo <= R:
            X = lo
            T = base + X
            pos = bisect_r(globB, T)
            if pos <= n - M:
                res[i] = -1
            else:
                r = pos + M - n
                L = pos - r
                if idx >= L:
                    S_eff = (pref[pos] - pref[L]) - globB[idx] + globB[L - 1]
                else:
                    S_eff = pref[pos] - pref[L]
                if (r + 1) * T > R + S_eff + base - r:
                    res[i] = X
                else:
                    res[i] = -1
        else:
            res[i] = -1

    sys.stdout.write(" ".join(str(x) for x in res))
    
if __name__ == '__main__':
    main()