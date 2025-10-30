#!/usr/bin/env python3
import sys
import threading
def main():
    import bisect
    input = sys.stdin.readline
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    S = sum(A)
    R = K - S  # remaining votes

    # Pair each (votes, index) and sort descending by votes
    ord_desc = sorted([(A[i], i) for i in range(N)], reverse=True)
    # Prepare for each i the list of its M strongest rivals
    # We only need their vote‐counts sorted descending, and their prefix sums (for fast range‐sum).
    # Precompute global top M+1
    topM1 = [v for v,_ in ord_desc[:M+1]]
    ps_topM1 = [0]
    for v in topM1:
        ps_topM1.append(ps_topM1[-1]+v)
    # Also global top M
    topM  = [v for v,_ in ord_desc[:M]]
    ps_topM = [0]
    for v in topM:
        ps_topM.append(ps_topM[-1]+v)

    res = [0]*N
    for rank, (votes_i, i) in enumerate(ord_desc):
        # Build L_i: if i not among global top M => rivals are topM
        # else rivals are topM1 except the one equal to i
        if rank>=M:
            L = topM
            ps = ps_topM
        else:
            # remove ord_desc[rank] from topM1
            # which is at position rank in ord_desc[:M+1]
            # so rivals are topM1[0:rank] + topM1[rank+1:M+1]
            # build array
            L = topM1[:rank] + topM1[rank+1:]
            # its prefix sums
            ps = [0]
            for v in L:
                ps.append(ps[-1]+v)
        # now L is size M, sorted descending
        # we will binary‐search X in [0..R]
        if M > N-1:
            # there aren't enough rivals to fill M slots => always wins
            res[i] = 0
            continue

        # helper: check whether X suffices
        def ok(X):
            T = votes_i + X
            # among L find how many have D <= T  => they are at the tail since L desc
            # i.e. find first index in L desc where v <= T
            # we can do: pos = bisect.bisect_right( [-v for v in L], -T )
            # but better linear transform: let L2 = reversed(L), now ascending
            # so pos2 = count of v2 <= T in L2 -> bisect_right
            # pos2 is t
            # We don't want to rebuild L2 each time; just compute by: in L sorted desc length M:
            # index p = first j in [0..M) with L[j] <= T  => p = bisect_left( [ -v for v in L ], -T )
            p = bisect.bisect_left(L, T, key=lambda x: -x) if False else None
            # Python's bisect doesn't accept key; do manual:
            lo, hi = 0, M
            while lo<hi:
                mid=(lo+hi)//2
                if L[mid] > T:
                    lo = mid+1
                else:
                    hi = mid
            p = lo
            t = M-p
            if t==0:
                # no rival needs votes => S=0
                Sneed = 0
            else:
                # sum of those t small ones is sum of L[p..M-1] = ps[M]-ps[p]
                sum_small = ps[M] - ps[p]
                # S = t*(T+1) - sum_small
                Sneed = t*(T+1) - sum_small
            # adversary has R-X votes to spend; if Sneed > R-X => cannot make M beaters => we win
            return Sneed > R - X

        # quick check: even with all R to i, can't win?
        if not ok(R):
            res[i] = -1
            continue
        # else binary search min X
        lo, hi = 0, R
        while lo<hi:
            mid = (lo+hi)//2
            if ok(mid):
                hi = mid
            else:
                lo = mid+1
        res[i] = lo

    print(" ".join(str(res[i]) for i in range(N)))

if __name__ == "__main__":
    main()