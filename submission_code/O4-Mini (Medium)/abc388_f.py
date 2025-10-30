import sys
import threading

def main():
    import sys
    import bisect

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M, A, B = map(int, input().split())
    bads = []
    for _ in range(M):
        l, r = map(int, input().split())
        bads.append((l, r))
    # build good segments as complement of bads in [1..N]
    goods = []
    prev = 1
    for (l, r) in bads:
        if prev <= l - 1:
            goods.append((prev, l - 1))
        prev = r + 1
    if prev <= N:
        goods.append((prev, N))
    # goods is list of (s,e), sorted, disjoint
    Gs = [seg[0] for seg in goods]
    Ge = [seg[1] for seg in goods]
    # reachable intervals R, as list of disjoint sorted intervals
    R = [(1, 1)]
    # frontier intervals F
    F = [(1, 1)]

    # function to subtract R from interval [l,r], return list of uncovered pieces
    def subtract_R(l, r, R):
        res = []
        cur = l
        # find first R index that may overlap
        idx = bisect.bisect_left(R, (l, -1))
        # check previous
        if idx > 0 and R[idx-1][1] >= l:
            j = idx - 1
        else:
            j = idx
        nR = len(R)
        # process overlapping Rs
        while j < nR and R[j][0] <= r:
            rl, rr = R[j]
            if rl > cur:
                res.append((cur, rl - 1))
            # move cur beyond this R interval
            if rr + 1 > cur:
                cur = rr + 1
            if cur > r:
                break
            j += 1
        if cur <= r:
            res.append((cur, r))
        return res

    # main loop
    while F:
        # collect candidate landing intervals
        Cand = []
        for (l, r) in F:
            l2 = l + A
            if l2 > N:
                continue
            r2 = r + B
            if r2 > N:
                r2 = N
            # find goods overlapping [l2, r2]
            # first good with end >= l2
            idx = bisect.bisect_left(Ge, l2)
            # that gives first i where Ge[i] >= l2
            # scan until goods start > r2
            gN = len(goods)
            j = idx
            while j < gN and Gs[j] <= r2:
                nl = l2 if l2 > Gs[j] else Gs[j]
                nr = r2 if r2 < Ge[j] else Ge[j]
                if nl <= nr:
                    Cand.append((nl, nr))
                j += 1
        if not Cand:
            break
        # merge Cand intervals
        Cand.sort()
        merged = []
        cl, cr = Cand[0]
        for l, r in Cand[1:]:
            if l <= cr + 1:
                if r > cr:
                    cr = r
            else:
                merged.append((cl, cr))
                cl, cr = l, r
        merged.append((cl, cr))
        CandM = merged

        # subtract R from CandM to get New intervals
        New = []
        for (l, r) in CandM:
            # subtract already reachable R
            parts = subtract_R(l, r, R)
            for (a, b) in parts:
                # check if N in [a,b]
                if a <= N <= b:
                    print("Yes")
                    return
                New.append((a, b))
        if not New:
            break
        # sort and merge New
        New.sort()
        merged = []
        cl, cr = New[0]
        for l, r in New[1:]:
            if l <= cr + 1:
                if r > cr:
                    cr = r
            else:
                merged.append((cl, cr))
                cl, cr = l, r
        merged.append((cl, cr))
        NewM = merged
        # merge NewM into R
        all_ints = R + NewM
        all_ints.sort()
        merged = []
        cl, cr = all_ints[0]
        for l, r in all_ints[1:]:
            if l <= cr + 1:
                if r > cr:
                    cr = r
            else:
                merged.append((cl, cr))
                cl, cr = l, r
        merged.append((cl, cr))
        R = merged
        # next frontier
        F = NewM

    # if loop ends without reaching N
    print("No")

if __name__ == "__main__":
    main()