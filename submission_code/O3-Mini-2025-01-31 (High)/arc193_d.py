def main():
    import sys
    data = sys.stdin.buffer.read().splitlines()
    if not data:
        return
    t = int(data[0])
    out_lines = []
    di = 1

    # This "feasible" function checks whether, given T, one can partition
    # the sorted list p (pieces’ positions) into r groups (one per required square Q)
    # such that every piece in the j-th group lies in [Q[j]-T, Q[j]+T].
    # We use a reservation constraint to guarantee that at least one piece is left for each future group.
    def feasible(T, p, Q, m, r):
        L = 0
        for j in range(r):
            low_bound = Q[j] - T
            high_bound = Q[j] + T
            # There must be at least one piece in the j-th group.
            if L >= m:
                return False
            # The first piece of the current group (p[L]) must lie between low_bound and high_bound.
            if p[L] < low_bound or p[L] > high_bound:
                return False
            # To "reserve" at least one piece for each later group,
            # the maximum index available for group j is
            max_idx = m - (r - j)
            cand = L
            # Advance cand (within [L, max_idx]) as long as next piece is within high_bound.
            while cand < max_idx and p[cand + 1] <= high_bound:
                cand += 1
            # Now assign group j to pieces p[L...cand] and update L.
            L = cand + 1
        return L == m

    # Process each test case:
    for _ in range(t):
        if di >= len(data):
            break
        try:
            n = int(data[di])
        except:
            break
        di += 1
        A = data[di].strip().decode()
        di += 1
        B = data[di].strip().decode()
        di += 1

        # Build lists: positions (1-indexed) with a piece in A and required squares in B.
        p = [i for i, ch in enumerate(A, start=1) if ch == '1']
        Q = [i for i, ch in enumerate(B, start=1) if ch == '1']
        m = len(p)
        r = len(Q)
        if m < r:
            out_lines.append("-1")
            continue

        # Binary search on T from 0 to n.
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) >> 1
            if feasible(mid, p, Q, m, r):
                hi = mid
            else:
                lo = mid + 1
        if feasible(lo, p, Q, m, r):
            out_lines.append(str(lo))
        else:
            out_lines.append("-1")
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()