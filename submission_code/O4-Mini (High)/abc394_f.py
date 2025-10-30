import sys
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.readline
    N_line = data()
    if not N_line:
        return
    N = int(N_line)
    adjT = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u,v = map(int, data().split())
        adjT[u].append(v)
        adjT[v].append(u)
    # find candidate internal nodes C: deg_T >=4
    isC = [False]*(N+1)
    nodesC = []
    for i in range(1, N+1):
        if len(adjT[i]) >= 4:
            isC[i] = True
            nodesC.append(i)
    # if no candidate, no alkane
    if not nodesC:
        print(-1)
        return
    # build adjacency among C-nodes
    adjC = [[] for _ in range(N+1)]
    for u in nodesC:
        for v in adjT[u]:
            if isC[v]:
                adjC[u].append(v)
    visitedC = [False]*(N+1)
    parent = [0]*(N+1)
    # children lists for each C-node
    children = [[] for _ in range(N+1)]
    # DP arrays
    dp_down1 = [0]*(N+1)  # dp when parent-edge used (cap = 3)
    dp_up1 = [0]*(N+1)    # dp contribution from parent side
    answer_k = 0

    # process each component in the C-forest
    for r in nodesC:
        if visitedC[r]:
            continue
        # DFS1 to build parent/children and postorder
        stack = [(r, 0, False)]
        postorder = []
        visitedC[r] = True  # mark root visited early
        parent[r] = 0
        # children lists for this comp will be filled
        while stack:
            u, p, done = stack.pop()
            if not done:
                # first time
                # we already marked visitedC[u] when pushing for the first time
                parent[u] = p
                stack.append((u, p, True))
                for v in adjC[u]:
                    if v == p or visitedC[v]:
                        continue
                    visitedC[v] = True
                    stack.append((v, u, False))
            else:
                # postorder position
                postorder.append(u)
                if p != 0:
                    children[p].append(u)
        # dp_down1: bottom-up
        for u in postorder:
            # pick top 3 dp_down1 among children[u]
            # we can find them by scanning
            m1 = m2 = m3 = 0
            for v in children[u]:
                val = dp_down1[v]
                if val > m1:
                    m3 = m2
                    m2 = m1
                    m1 = val
                elif val > m2:
                    m3 = m2
                    m2 = val
                elif val > m3:
                    m3 = val
            dp_down1[u] = 1 + m1 + m2 + m3

        # dp_up1 for root is zero (no parent side)
        dp_up1[r] = 0

        # reroot DP: compute dp_up1 for children and compute dp0_entire (cap=4) for each u
        stack2 = [r]
        while stack2:
            u = stack2.pop()
            p = parent[u]
            # build neighbor contributions f(u,w)
            # from children
            degu = 0
            # we'll build a small list of (value, id)
            # using manual top-4 selection
            top = []  # will store up to 4 pairs sorted by value desc
            # helper to insert (val,id) into top-4
            def try_insert(pair):
                # pair = (val, id)
                if len(top) < 4:
                    top.append(pair)
                    if len(top) == 4:
                        # sort descending by val
                        top.sort(key=lambda x: x[0], reverse=True)
                else:
                    # top is sorted desc, top[3] is 4th largest
                    if pair[0] > top[3][0]:
                        # find position to insert
                        fv = pair[0]
                        # linear scan to find insertion pos
                        for ii in range(4):
                            if fv > top[ii][0]:
                                top.insert(ii, pair)
                                top.pop()  # keep size 4
                                break

            # from children
            for v in children[u]:
                try_insert((dp_down1[v], v))
            # from parent side
            if p != 0:
                # dp_up1[u] contribution from parent
                try_insert((dp_up1[u], p))

            # now top holds up to 4 best (value,id), sorted desc
            # compute sum of top-4 and top-3
            sum4 = 0
            sum3 = 0
            tlen = len(top)
            # already sorted if len(top)==4; if less, may be unsorted, so sort now
            if tlen > 1:
                # sort top by value desc for correct prefix sums and index
                top.sort(key=lambda x: x[0], reverse=True)
            for i in range(tlen):
                if i < 4:
                    sum4 += top[i][0]
                if i < 3:
                    sum3 += top[i][0]
            # dp0_entire[u] = dp when root-edge-used=0, cap=4 children
            k_u = 1 + sum4
            if k_u > answer_k:
                answer_k = k_u
            # compute dp_up1 for each child v
            # for each child v, exclude its pair from top-3 selection
            # top is sorted desc
            for v in children[u]:
                fv = dp_down1[v]
                # find position idx of v in top (only if in top)
                idx = -1
                # we only care if it's among the first 3 (positions 0,1,2)
                # but we need its exact pos if in top
                for j in range(tlen):
                    if top[j][1] == v:
                        idx = j
                        break
                if idx >= 0 and idx < 3:
                    # its value contributed to sum3
                    if tlen > 3:
                        # there is a replacement candidate at top[3]
                        sumExcl3 = sum3 - fv + top[3][0]
                    else:
                        # fewer than 4 neighbors, just subtract
                        sumExcl3 = sum3 - fv
                else:
                    # either not in top or in position >=3
                    sumExcl3 = sum3
                dp_up1[v] = 1 + sumExcl3
                stack2.append(v)
        # end reroot for this component
    # end for each component

    # answer_k is max number of internals
    # total vertices in alkane = 3*k + 2
    result = 3 * answer_k + 2
    print(result)

if __name__ == "__main__":
    main()