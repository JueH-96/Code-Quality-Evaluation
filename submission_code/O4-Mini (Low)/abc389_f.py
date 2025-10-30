import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    input = sys.stdin.readline

    N = int(input())
    contests = [tuple(map(int, input().split())) for _ in range(N)]
    Q = int(input())
    queries = [int(input()) for _ in range(Q)]
    MAXR = 500000

    # For each starting rating x, we want to know how many contests
    # will increment the rating when run in order.  Direct per-query
    # simulation is too slow.  Instead we precompute for every x
    # its final rating f[x], for x=1..MAXR, in O(MAXR + N).

    # We maintain an array f of length MAXR+2, initially f[x]=x.
    # We'll process contests one by one; in each contest [L,R], we
    # want to increment f[x] by 1 for exactly those x where the
    # current f[x] (after previous contests) lies in [L,R].
    #
    # Observe that after k contests f[x] = x + d[x], where d[x] is
    # the number of those contests that covered the evolving rating.
    # The condition "current rating f[x] in [L,R]" is
    #   x + d[x] in [L,R]  <=>  d[x] in [L - x, R - x].
    # But d[x] is small (<=N), and R-x >= 0 only when x <= R.
    #
    # We can carry an array D of size MAXR+2, D[x]=d[x], and we
    # need for each contest to do:
    #   for x in [1..MAXR]:
    #     if L <= x + D[x] <= R:  D[x] += 1
    # Doing that naively is O(MAXR) per contest.  Too slow.
    #
    # However, x + D[x] is strictly increasing in x if D[x] grows
    # slowly.  We can use a two‐pointer sweep to do each contest
    # in O(1) amortized by keeping track of intervals of x that
    # currently map to contiguous blocks [a..b] of ratings.
    #
    # Implementation:
    # Maintain a list of disjoint segments over x where x + D[x]
    # is constant offset by x.  Actually we track the mapping
    #    x -> y = x + D[x]
    # as a set of disjoint intervals on x that map onto contiguous
    # intervals on y with slope 1.
    #
    # Initially, D[x]=0 so x-> x, i.e. one segment [1..MAXR] -> [1..MAXR].
    # On contest [L,R], every x whose current y=x+D[x] falls in [L,R]
    # will have D[x] increased by 1; hence y increases by 1.
    #
    # So we must take the union of those pieces of the domain whose
    # images intersect [L,R], cut them out, shift them up by 1 in the
    # image, and re‐insert, merging with neighbors if possible.
    #
    # We can store these segments in a sorted list (by their image range).
    # Each segment is (lx, rx, ly) meaning x in [lx..rx] maps to
    # y in [ly..ly+(rx-lx)] by y = x - lx + ly.
    #
    # For a contest [L,R]:
    # 1. Find all segments whose image-range [ly..ry] intersects [L,R].
    # 2. For each such segment, we may need to split it into three:
    #    left-part (below L), mid-part (in [L,R]), right-part (above R).
    # 3. The mid-parts all get their ly increased by +1.
    # 4. Re‐merge adjacent segments if they become contiguous in x
    #    and y after the shift.
    #
    # Since across all contests each boundary is split/merged O(1)
    # times, the total is O((N + MAXR) log (N+MAXR)).  With MAXR=5e5,
    # it's OK in PyPy/CPython if we code carefully with bisect.

    import bisect

    # We'll keep segments sorted by their ly (image low end).
    # segs = list of [lx, rx, ly]
    segs = [[1, MAXR, 1]]
    # Also maintain a parallel list of the segment-image-starts for bisect
    keys = [1]  # the ly's

    def split_segment(idx, cut_lo, cut_hi):
        # Given segs[idx] = [lx, rx, ly], and we know
        # its image-range [ly..ry] = [ly..ly+(rx-lx)]
        # we want to cut out the slice where y in [cut_lo..cut_hi].
        # That slice in x-space is x in
        #   [cut_lo - ly + lx .. cut_hi - ly + lx].
        lx, rx, ly = segs[idx]
        length = rx - lx
        ry = ly + length
        lo = max(cut_lo, ly)
        hi = min(cut_hi, ry)
        if lo > hi:
            return []  # no overlap
        # Map back to x-range
        a = lx + (lo - ly)
        b = lx + (hi - ly)
        parts = []
        # Left piece [lx .. a-1]
        if a > lx:
            parts.append([lx, a-1, ly])
        # Middle piece [a..b]
        parts.append([a, b, lo])
        # Right piece [b+1..rx]
        if b < rx:
            parts.append([b+1, rx, lo + (b+1 - a)])
        return parts

    for (L, R) in contests:
        # Find first segment whose ry >= L
        # We search by ly, but need to scan a little
        i = bisect.bisect_right(keys, R)  # first seg with ly > R
        j = bisect.bisect_left(keys, L)   # first seg with ly >= L
        # Actually segments with ly <= R might still start below L,
        # so we must walk back one if possible
        if j > 0:
            j -= 1
        # Now scan forward from j while segments overlap [L,R]
        newsegs = []
        to_remove = []
        k = j
        while k < len(segs):
            lx, rx, ly = segs[k]
            ry = ly + (rx - lx)
            if ly > R:
                break
            if ry < L:
                k += 1
                continue
            # This segment overlaps [L,R] in the image
            parts = split_segment(k, L, R)
            to_remove.append(k)
            for p in parts:
                newsegs.append(p)
            k += 1
        # Remove old
        for idx in reversed(to_remove):
            segs.pop(idx)
            keys.pop(idx)
        # Fix the middle parts that lie in [L,R] -> shift up by +1 in y
        for p in newsegs:
            lx, rx, ly = p
            # If its image [ly..] was fully inside [L,R], that was the 'mid'
            # which we detect by ly>=L && ly+(rx-lx)<=R
            if L <= ly and ly + (rx-lx) <= R:
                p[2] += 1
        # Insert back all new
        for p in newsegs:
            ly = p[2]
            idx = bisect.bisect_left(keys, ly)
            segs.insert(idx, p)
            keys.insert(idx, ly)
        # Merge neighbors
        m = 0
        merged = []
        while m < len(segs):
            lx, rx, ly = segs[m]
            m += 1
            while m < len(segs):
                lx2, rx2, ly2 = segs[m]
                # can merge if lx2==rx+1 and ly2==ly+(rx-lx+1)
                if lx2 == rx+1 and ly2 == ly + (rx-lx+1):
                    rx = rx2
                    m += 1
                else:
                    break
            merged.append([lx, rx, ly])
        segs = merged
        keys = [s[2] for s in segs]

    # Now segs partitions [1..MAXR] in x, and for x in [lx..rx] we have
    # final f[x] = ly + (x-lx).  Answer queries in O(log segs).
    out = []
    for x in queries:
        i = bisect.bisect_right(keys, x) - 1
        lx, rx, ly = segs[i]
        res = ly + (x - lx)
        out.append(str(res))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()