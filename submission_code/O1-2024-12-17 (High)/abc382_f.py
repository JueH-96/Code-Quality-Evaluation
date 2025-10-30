def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    # Fast parsing of input
    H, W, N = map(int, input_data[:3])
    bars_raw = input_data[3:]
    bars = []
    idx = 0
    for i in range(N):
        R = int(bars_raw[idx]); C = int(bars_raw[idx+1]); L = int(bars_raw[idx+2])
        idx += 3
        s = C
        e = C + L - 1
        # Store (row, start_col, end_col, original_index)
        bars.append((R, s, e, i))
    # Sort bars by descending row (larger R first).
    # If two bars have the same R, any order among them is fine.
    bars.sort(key=lambda x: x[0], reverse=True)

    # We will perform a "segment tree" with lazy propagation
    # for range-max queries and range-max updates on [1..W].
    # Then for each bar (in descending R), we do:
    #   current_max = query_max(s, e)
    #   L_i = current_max+1
    #   update_range(s, e, L_i)
    #
    # Finally R'_i = H+1 - L_i.

    # Segment-tree (and lazy) arrays (1-based indexing for tree nodes)
    # Size ~ 4 * W is safe.
    size_tree = 1
    while size_tree < W:
        size_tree <<= 1
    seg = [0] * (size_tree << 1)
    lazy = [0] * (size_tree << 1)

    def push_down(idx: int, left: int, right: int):
        # Push lazy updates to children, if any
        if lazy[idx] != 0:
            mid = (left + right) >> 1
            lc = idx << 1
            rc = lc | 1
            # apply to left child
            seg[lc] = max(seg[lc], lazy[idx])
            lazy[lc] = max(lazy[lc], lazy[idx])
            # apply to right child
            seg[rc] = max(seg[rc], lazy[idx])
            lazy[rc] = max(lazy[rc], lazy[idx])
            lazy[idx] = 0

    def update_range(idx: int, start: int, end: int, ql: int, qr: int, val: int):
        if ql > end or qr < start:
            # no overlap
            return
        if ql <= start and end <= qr:
            # fully covered
            seg[idx] = max(seg[idx], val)
            lazy[idx] = max(lazy[idx], val)
            return
        # partial coverage
        push_down(idx, start, end)
        mid = (start + end) >> 1
        update_range(idx << 1, start, mid, ql, qr, val)
        update_range((idx << 1) | 1, mid+1, end, ql, qr, val)
        seg[idx] = max(seg[idx << 1], seg[(idx << 1) | 1])

    def query_range(idx: int, start: int, end: int, ql: int, qr: int) -> int:
        if ql > end or qr < start:
            return 0
        if ql <= start and end <= qr:
            return seg[idx]
        push_down(idx, start, end)
        mid = (start + end) >> 1
        return max(
            query_range(idx << 1, start, mid, ql, qr),
            query_range((idx << 1) | 1, mid+1, end, ql, qr)
        )

    # Array to store the "chain length" L_i for each bar i
    Lvals = [0]*N

    # Process bars in descending order of R
    for R, s, e, i in bars:
        # Range max query on [s..e]
        cur_max = query_range(1, 1, size_tree, s, e)
        Lvals[i] = cur_max + 1
        # Range update: set max in [s..e] to Lvals[i]
        update_range(1, 1, size_tree, s, e, Lvals[i])

    # Compute final row = H+1 - L_i
    # Then output in the order of bar indices 1..N
    # (which in code is just i=0..N-1).
    out = []
    for i in range(N):
        out.append(str(H + 1 - Lvals[i]))
    print("
".join(out))


# Do not forget to call main()
if __name__ == "__main__":
    main()