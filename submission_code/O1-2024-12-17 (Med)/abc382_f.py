def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    # parse inputs
    H, W, N = map(int, input_data[:3])
    bars = []
    idx = 3
    for i in range(N):
        r = int(input_data[idx]); c = int(input_data[idx+1]); l = int(input_data[idx+2])
        idx += 3
        # store (R_i, C_i, L_i, original_index)
        bars.append((r, c, l, i))

    # We will sort bars in descending order of R_i (larger R_i first).
    # Ties (if any) can be broken by descending order of i just to have a consistent ordering,
    # though it generally does not affect correctness if we handle them in any fixed order
    # when R_i are equal.
    # Python's sort is stable, so we can sort by bar index descending then R descending:
    # or simply sort by (R_i, i) descending in one pass:
    bars.sort(key=lambda x: (x[0], x[3]), reverse=True)

    # We need a segment tree (or Fenwick) that supports:
    # 1. range-min query
    # 2. range update: occupant[c] = min(occupant[c], val) for c in [L..R]
    #
    # We'll implement a lazy segment tree.  Each leaf covers one column,
    # and the value stored is the minimum occupant row "from the top" 
    # (initially H+1 means no occupant).
    # For a bar i that spans [C..C+L-1], we find occupantRangeMin = rangeMinQuery(C, C+L-1).
    # Then final_row = max(R_i, occupantRangeMin - 1).
    # (and clamp to ≤ H just in case, though occupantRangeMin-1 should never exceed H if occupantRangeMin ≤ H+1).
    # Then we do rangeMinUpdate(C, C+L-1, final_row).
    #
    # The final rows are stored in an array res[i] = final_row, where i is the original bar index.

    # Build the segment tree data structures:
    import math
    size = 1
    while size < W:
        size <<= 1

    INF = H + 1  # occupant "no block" initialization

    # tree will hold the segment values (minimum occupant in that segment)
    # lazy will hold the "proposed min" updates that need to be pushed down.
    tree = [INF]*(2*size)
    lazy = [INF]*(2*size)

    # The build is trivial since everything is initially INF.
    # define push_down and push_up methods to handle lazy propagation of "min"-updates.

    def apply_set(p, val):
        # When applying a "min update" to node p, we set both the tree value and lazy tag to min of current and val
        tree[p] = min(tree[p], val)
        lazy[p] = min(lazy[p], val)

    def push_down(p):
        # push lazy value down to children
        if lazy[p] < INF:
            apply_set(p<<1, lazy[p])
            apply_set(p<<1|1, lazy[p])
            lazy[p] = INF

    def pull_up(p):
        # recalc this node from children
        tree[p] = min(tree[p<<1], tree[p<<1|1])

    def range_min_update(l, r, val):
        # update occupant[x] = min(occupant[x], val) for x in [l..r]
        # internal recursive or iterative segment tree approach
        def _update(start, end, idx, left, right, value):
            if right < start or end < left:
                return
            if left <= start and end <= right:
                apply_set(idx, value)
                return
            mid = (start+end)//2
            push_down(idx)
            _update(start, mid, idx<<1, left, right, value)
            _update(mid+1, end, idx<<1|1, left, right, value)
            pull_up(idx)

        _update(1, size, 1, l, r, val)


    def range_min_query(l, r):
        # returns min value in [l..r]
        # iterative version
        resv = INF
        def _query(start, end, idx, left, right):
            nonlocal resv
            if right < start or end < left:
                return
            if left <= start and end <= right:
                resv = min(resv, tree[idx])
                return
            push_down(idx)
            mid = (start+end)//2
            _query(start, mid, idx<<1, left, right)
            _query(mid+1, end, idx<<1|1, left, right)

        _query(1, size, 1, l, r)
        return resv

    # We'll store the final row for each bar:
    res = [0]*N

    # Process bars in descending order of R
    for (r, c, length, idx_original) in bars:
        L = c
        Rcol = c + length - 1
        # get min occupant in [L..Rcol]
        occ_min = range_min_query(L, Rcol)
        # bar wants to occupy occupant_range_min - 1 if that is >= r
        final_row = max(r, occ_min - 1)
        if final_row > H:  # clamp to H if somehow it overshoots
            final_row = H
        res[idx_original] = final_row
        # now update occupant in [L..Rcol] to final_row
        range_min_update(L, Rcol, final_row)

    # Finally, print results in the original order:
    print('
'.join(map(str, res)))


# do not forget to call main()!
if __name__ == "__main__":
    main()