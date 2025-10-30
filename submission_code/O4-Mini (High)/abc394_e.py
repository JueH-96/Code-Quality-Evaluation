import sys
def main():
    import sys
    sys.setrecursionlimit(1000000)
    input = sys.stdin.readline

    N_line = input().strip()
    if not N_line:
        return
    N = int(N_line)
    labels = [input().rstrip() for _ in range(N)]
    # in_by_label_list[u'][c] = list of u such that u->u' label c
    in_by_label_list = [ [ [] for _ in range(26) ] for _ in range(N) ]
    # out_mask_bit[c][v'] = bitmask of v such that v'->v label c
    out_mask_bit = [ [0]*N for _ in range(26) ]
    # mask of any outgoing edge per u for BFS_odd initial
    out_edge_mask_all = [0]*N

    for u in range(N):
        row = labels[u]
        mask_row = 0
        for v, ch in enumerate(row):
            if ch != '-':
                c = ord(ch) - 97
                # record in-edge for v
                in_by_label_list[v][c].append(u)
                # record out-edge bitmask for u
                out_mask_bit[c][u] |= (1 << v)
                mask_row |= (1 << v)
        out_edge_mask_all[u] = mask_row

    # Precompute for each u' the list of labels c for which in_by_label_list[u'][c] non-empty
    in_labels = [ [c for c in range(26) if in_by_label_list[u][c]] for u in range(N) ]

    # neighbor generation for reversed BFS
    def neighbor_gen(f_rows):
        # f_rows: list of ints length N, bitmask per u' of v' in frontier
        f_new = [0]*N
        # local references
        in_labels_local = in_labels
        in_by_label_local = in_by_label_list
        out_mask_local = out_mask_bit
        for up in range(N):
            row_mask = f_rows[up]
            if row_mask == 0:
                continue
            # for each label c with incoming edges into up
            for c in in_labels_local[up]:
                in_us = in_by_label_local[up][c]
                # compute union of all out-masks for v' bits
                tmp = row_mask
                out_v_mask = 0
                # iterate bits in tmp
                while tmp:
                    lsb = tmp & -tmp
                    vp = lsb.bit_length() - 1
                    out_v_mask |= out_mask_local[c][vp]
                    tmp ^= lsb
                if out_v_mask:
                    # set these v for all u in in_us
                    for u in in_us:
                        f_new[u] |= out_v_mask
        return f_new

    # BFS for even-length pal (pair transitions) reversed
    dist_even = [ [-1]*N for _ in range(N) ]
    visited_even = [0]*N
    f_rows_even = [0]*N
    # initial: (u,u) for all u
    for u in range(N):
        visited_even[u] = (1 << u)
        f_rows_even[u] = (1 << u)
        dist_even[u][u] = 0
    d = 0
    ng = neighbor_gen
    # BFS layers
    while True:
        f_next = ng(f_rows_even)
        any_new = False
        new_rows = [0]*N
        # filter out already visited
        for u in range(N):
            # new bits are those in f_next but not in visited_even
            new_mask = f_next[u] & ~visited_even[u]
            if new_mask:
                any_new = True
                new_rows[u] = new_mask
        if not any_new:
            break
        d += 1
        # record distances and mark visited
        for u in range(N):
            m = new_rows[u]
            if m:
                visited_even[u] |= m
                tmp = m
                # assign dist_even[u][v] = d
                while tmp:
                    lsb = tmp & -tmp
                    v = lsb.bit_length() - 1
                    dist_even[u][v] = d
                    tmp ^= lsb
        f_rows_even = new_rows

    # BFS for odd-length pal (one central edge + pair transitions) reversed
    dist_odd = [ [-1]*N for _ in range(N) ]
    visited_odd = [0]*N
    # initial f_rows_odd = all (u,v) where u->v exists
    f_rows_odd = out_edge_mask_all[:]  # copy
    # initialize visited and dist_odd at d=0
    for u in range(N):
        m = f_rows_odd[u]
        if m:
            visited_odd[u] = m
            tmp = m
            while tmp:
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                dist_odd[u][v] = 0
                tmp ^= lsb
    d = 0
    # BFS layers
    while True:
        f_next = ng(f_rows_odd)
        any_new = False
        new_rows = [0]*N
        for u in range(N):
            new_mask = f_next[u] & ~visited_odd[u]
            if new_mask:
                any_new = True
                new_rows[u] = new_mask
        if not any_new:
            break
        d += 1
        for u in range(N):
            m = new_rows[u]
            if m:
                visited_odd[u] |= m
                tmp = m
                while tmp:
                    lsb = tmp & -tmp
                    v = lsb.bit_length() - 1
                    dist_odd[u][v] = d
                    tmp ^= lsb
        f_rows_odd = new_rows

    out = []
    INF = 10**18
    # build output
    for u in range(N):
        row_ans = []
        for v in range(N):
            best = INF
            de = dist_even[u][v]
            if de >= 0:
                val = de * 2
                if val < best:
                    best = val
            do = dist_odd[u][v]
            if do >= 0:
                val = do * 2 + 1
                if val < best:
                    best = val
            if best == INF:
                row_ans.append("-1")
            else:
                row_ans.append(str(best))
        out.append(" ".join(row_ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()