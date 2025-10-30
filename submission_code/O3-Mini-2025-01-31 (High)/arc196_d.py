# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it)); M = int(next(it)); Q = int(next(it))
    # For each journey i (person i) compute:
    #   l = min(S, T) and r = max(S, T)
    #   typ = 1 if S < T ("up" journey) and 0 if S > T ("down" journey)
    journeys = []  # each entry: (l, r, typ, i) with i = journey number (1-indexed)
    for i in range(1, M+1):
        s = int(next(it))
        t = int(next(it))
        if s < t:
            l, r, typ = s, t, 1
        else:
            l, r, typ = t, s, 0
        journeys.append((l, r, typ, i))
    
    # Separate journeys by type.
    up_ints = []    # for typ == 1
    down_ints = []  # for typ == 0
    for (l, r, typ, idx) in journeys:
        if typ == 1:
            up_ints.append((l, r, idx))
        else:
            down_ints.append((l, r, idx))
    
    # We now “record a conflict” by preparing an array conflict_next[0..M-1] (for journey 1..M)
    # Initialize with INF.
    INF = M + 1
    conflict_next = [INF] * M  # conflict_next[i] is for journey (i+1)

    # For journeys of the same type, a standard necessary condition is that:
    # if two journeys (with intervals [l,r]) overlap then the later–journey MUST be nested;
    # that is, if sorted by l then for overlapping intervals, we require that the later interval’s r <= earlier interval’s r.
    # Also if two journeys share the same left endpoint then they must have equal r.
    # We “detect” a violation using a stack.
    
    # Process up–journeys (typ == 1)
    # sort by (l, -r): so if same l the one with larger r comes first.
    up_ints.sort(key=lambda x: (x[0], -x[1]))
    stack = []
    for (l, r, idx) in up_ints:
        while stack and stack[-1][1] <= l:
            stack.pop()
        if stack and l < stack[-1][1]:
            top_l, top_r, top_idx = stack[-1]
            # For valid nesting we need: r <= top_r.
            # Also if left endpoints are equal then they must be equal.
            if (r > top_r) or (r == top_r and l != top_l):
                # record conflict of the pair (top_idx, idx); choose smaller index as representative.
                x = top_idx if top_idx < idx else idx
                y = top_idx if top_idx > idx else idx
                if x - 1 >= 0 and y < conflict_next[x-1]:
                    conflict_next[x-1] = y
        stack.append((l, r, idx))
    
    # Process down–journeys in the same way.
    down_ints.sort(key=lambda x: (x[0], -x[1]))
    stack = []
    for (l, r, idx) in down_ints:
        while stack and stack[-1][1] <= l:
            stack.pop()
        if stack and l < stack[-1][1]:
            top_l, top_r, top_idx = stack[-1]
            if (r > top_r) or (r == top_r and l != top_l):
                x = top_idx if top_idx < idx else idx
                y = top_idx if top_idx > idx else idx
                if x - 1 >= 0 and y < conflict_next[x-1]:
                    conflict_next[x-1] = y
        stack.append((l, r, idx))
    
    # Next, for journeys that share exactly the same endpoints (l,r) but occur in opposite types,
    # a conflict appears (since one “requires” the interior to be above the end–value, and the other requires it to be below).
    same = {}  # key: (l, r) -> [min_up, max_up, min_down, max_down]
    for (l, r, typ, idx) in journeys:
        key = (l, r)
        if key not in same:
            same[key] = [INF, 0, INF, 0]
        if typ == 1:
            if idx < same[key][0]:
                same[key][0] = idx
            if idx > same[key][1]:
                same[key][1] = idx
        else:
            if idx < same[key][2]:
                same[key][2] = idx
            if idx > same[key][3]:
                same[key][3] = idx
    for key, vals in same.items():
        min_up, max_up, min_down, max_down = vals
        if min_up != INF and min_down != INF:
            x = min(min_up, min_down)
            y = max(max_up, max_down)
            if x - 1 >= 0 and y < conflict_next[x-1]:
                conflict_next[x-1] = y

    # Now, answering queries.
    # (A set of journeys (with input indices L .. R) is feasible if and only if
    #  the minimum conflict_next among journeys in the block (i.e. 
    #  min{ conflict_next[i] for i from L–1 to R–1 } ) is > R.)
    #
    # For fast queries we build an RMQ structure (segment tree) on conflict_next.
    size = 1
    while size < M:
        size *= 2
    seg = [INF] * (2 * size)
    for i in range(M):
        seg[size + i] = conflict_next[i]
    for i in range(M, size):
        seg[size + i] = INF
    for i in range(size-1, 0, -1):
        a = seg[2*i]
        b = seg[2*i+1]
        seg[i] = a if a < b else b

    def seg_query(l, r):
        res = INF
        l += size
        r += size
        while l <= r:
            if l & 1:
                if seg[l] < res:
                    res = seg[l]
                l += 1
            if (r & 1) == 0:
                if seg[r] < res:
                    res = seg[r]
                r -= 1
            l //= 2
            r //= 2
        return res

    # Process queries.
    out_lines = []
    for _ in range(Q):
        Lq = int(next(it))
        Rq = int(next(it))
        # Convert to 0-indexed.
        lq = Lq - 1
        rq = Rq - 1
        mn = seg_query(lq, rq)
        # If some journey i in [Lq,Rq] has conflict partner j <= Rq then a forbidden pair is contained.
        if mn <= Rq:
            out_lines.append("No")
        else:
            out_lines.append("Yes")
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()