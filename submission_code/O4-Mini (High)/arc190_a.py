import sys
import threading
def main():
    import sys
    import threading
    data = sys.stdin

    line = data.readline().split()
    if not line:
        return
    N = int(line[0]); M = int(line[1])
    Ls = [0]*M
    Rs = [0]*M
    intervals = []
    full_idx = -1
    Lmax = -1; id_lmax = -1
    Rmin = 10**9+1; id_rmin = -1

    for i in range(M):
        parts = data.readline().split()
        l = int(parts[0]); r = int(parts[1])
        Ls[i] = l; Rs[i] = r
        intervals.append((l, r, i))
        # full cover detection
        if l == 1 and r == N:
            full_idx = i
        # track global max L and min R
        if l > Lmax:
            Lmax = l; id_lmax = i
        if r < Rmin:
            Rmin = r; id_rmin = i

    # Build sorted by L
    intervals.sort(key=lambda x: (x[0], x[1]))
    sorted_L = intervals  # list of (L,R,idx)

    # detect disjoint pair and nested pair
    dis_i = dis_j = -1
    nested_i = nested_j = -1
    prefix_min_R = 10**9+5
    prefix_min_idx = -1
    prefix_max_R = -1
    prefix_max_idx = -1

    for (l, r, idx) in sorted_L:
        # disjoint: any earlier interval with R < l
        if prefix_min_R < l and dis_i < 0:
            # prefix_min_idx interval and current idx are disjoint
            dis_i = prefix_min_idx
            dis_j = idx
        # nested: any earlier interval with R >= r contains current
        if prefix_max_R >= r and nested_i < 0:
            # prefix_max_idx covers current idx
            nested_i = idx
            nested_j = prefix_max_idx
        # update prefix_min_R
        if r < prefix_min_R:
            prefix_min_R = r
            prefix_min_idx = idx
        # update prefix_max_R
        if r > prefix_max_R:
            prefix_max_R = r
            prefix_max_idx = idx

    # greedy cover function using heap
    import heapq
    def greedy_cover(a, b, exclude_set):
        # cover [a,b] with minimal intervals from sorted_L, skipping exclude_set
        if a > b:
            return []
        ans = []
        pos = a
        n = M
        heap = []
        idx = 0
        # We will scan sorted_L; for each pos, push all intervals with L <= pos
        while pos <= b:
            # push intervals starting at or before pos
            while idx < n and sorted_L[idx][0] <= pos:
                l2, r2, id2 = sorted_L[idx]
                if exclude_set is None or id2 not in exclude_set:
                    # push with key = -r2
                    heapq.heappush(heap, (-r2, id2))
                idx += 1
            # no candidates cover pos
            if not heap:
                return None
            # pick best reach
            r_top, id_top = heapq.heappop(heap)
            bestR = -r_top
            if bestR < pos:
                return None
            ans.append(id_top)
            pos = bestR + 1
        return ans

    INF = 10**18
    bestCost = INF
    bestS1 = []
    bestS2 = set()

    # Case cost 1: single op1 on full interval
    if full_idx >= 0:
        print(1)
        res = ['0'] * M
        res[full_idx] = '1'
        print(' '.join(res))
        return

    # Case costCoverFull = greedy op1-only on [1,N]
    cover_full = greedy_cover(1, N, None)
    if cover_full is not None:
        c0 = len(cover_full)
    else:
        c0 = INF

    # record op1-only candidate
    if c0 < bestCost:
        bestCost = c0
        bestS1 = cover_full[:] if cover_full is not None else []
        bestS2 = set()

    # Case cost 2 via disjoint op2-only
    if dis_i >= 0:
        # cost 2
        if 2 < bestCost:
            bestCost = 2
            bestS1 = []
            bestS2 = {dis_i, dis_j}
    # Case cost 2 via nested op2+op1
    if nested_i >= 0:
        if 2 < bestCost:
            bestCost = 2
            bestS1 = [nested_j]
            bestS2 = {nested_i}
    # Case cost 2 via two op1 cover [1,N]
    if cover_full is not None and len(cover_full) == 2:
        if 2 < bestCost:
            bestCost = 2
            bestS1 = cover_full[:]
            bestS2 = set()

    # If bestCost is 2 already, we have optimal (since 1 not found)
    if bestCost == 2:
        # output
        print(2)
        res = ['0'] * M
        for i in bestS1:
            res[i] = '1'
        for i in bestS2:
            res[i] = '2'
        print(' '.join(res))
        return

    # Now bestCost >=3 or INF. Try K=2 combos J_all
    # S2 candidates: id_lmax, id_rmin if distinct
    if id_lmax >= 0 and id_rmin >= 0 and id_lmax != id_rmin:
        a2 = Lmax; b2 = Rmin
        # only if segment not full covered by nested above
        # cover J_all excluding id_lmax and id_rmin
        cover_Jall = greedy_cover(a2, b2, {id_lmax, id_rmin})
        if cover_Jall is not None:
            cost2 = 2 + len(cover_Jall)
            if cost2 < bestCost:
                bestCost = cost2
                bestS1 = cover_Jall[:]
                bestS2 = {id_lmax, id_rmin}

    # Now K=1 combos: try op2 on id_lmax and id_rmin
    # candidate i's = {id_lmax, id_rmin}
    cand = []
    if id_lmax >= 0:
        cand.append(id_lmax)
    if id_rmin >= 0:
        cand.append(id_rmin)
    # unique
    cand = list(dict.fromkeys(cand))
    for i in cand:
        a1 = Ls[i]; b1 = Rs[i]
        cover_Ji = greedy_cover(a1, b1, {i})
        if cover_Ji is not None:
            cost1 = 1 + len(cover_Ji)
            if cost1 < bestCost:
                bestCost = cost1
                bestS1 = cover_Ji[:]
                bestS2 = {i}

    # Finally, consider if initial cover_full (op1-only) gives bestCost but is INF => no solution
    # Also cover_full might be >3 but bestCost maybe INF or >=3
    # If bestCost still INF, no solution
    if bestCost >= INF//2:
        print(-1)
        return

    # Also consider initial op1-only if bestCost==c0 may be >=3
    # But bestCost holds that

    # Output bestCost and assignments
    print(bestCost)
    res = ['0'] * M
    for i in bestS1:
        res[i] = '1'
    for i in bestS2:
        res[i] = '2'
    print(' '.join(res))


if __name__ == "__main__":
    main()