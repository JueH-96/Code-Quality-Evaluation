def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    mod = 998244353

    #----------------------------------------------------------------
    # 1) Read input
    #----------------------------------------------------------------
    N = int(input_data[0])
    Q = int(input_data[1])
    p = [0]*Q
    v = [0]*Q
    idx = 2
    for i in range(Q):
        p[i] = int(input_data[idx]);   v[i] = int(input_data[idx+1])
        idx += 2

    #----------------------------------------------------------------
    # 2) Quick check: If two operations share the same p-value but come
    #    in increasing time order with a strictly smaller v afterward,
    #    that is impossible (they would overlap with bigger→smaller value).
    #    Concretely, for the same p, v must be non-decreasing in time order.
    #----------------------------------------------------------------
    last_val_for_p = {}
    for i in range(Q):
        pi, vi = p[i], v[i]
        if pi in last_val_for_p:
            if vi < last_val_for_p[pi]:
                # Impossible
                print(0)
                return
        last_val_for_p[pi] = vi

    #----------------------------------------------------------------
    # We will use 2-SAT on Q variables x_i ∈ {0,1},
    # where x_i=0 ⇒ operation i uses prefix, x_i=1 ⇒ operation i uses suffix.
    #
    # "No crying" means: if i<j and v[i]>v[j], then operation i and j
    # must not overlap.  The intervals overlap unless we pick one of the
    # "non-overlapping" combinations.  In particular:
    #  - If p[i] == p[j] and v[i]>v[j], impossible.
    #  - If p[i] <  p[j] and v[i]>v[j], we force (i→suffix, j→prefix).
    #  - If p[i] >  p[j] and v[i]>v[j], we force (i→prefix, j→suffix).
    #
    # We'll build an implication graph on 2*Q nodes: node (i*2 + b)
    # represents "x_i = b".  Forcing x_i=1 means x_i=0 => contradiction,
    # which is an edge (i0 -> i1).  Similarly forcing x_i=0 is (i1 -> i0).
    #----------------------------------------------------------------

    # For convenience define a small helper:
    def i0(i):  # "x_i = 0" node
        return 2*i
    def i1(i):  # "x_i = 1" node
        return 2*i+1

    adjacency = [[] for _ in range(2*Q)]

    #----------------------------------------------------------------
    # 3) Build edges in O(Q^2).  If v[i]>v[j] for i<j:
    #    - if p[i]==p[j] => no solution
    #    - if p[i]<p[j] => x_i=1, x_j=0 => edges (i0->i1), (j1->j0)
    #    - if p[i]>p[j] => x_i=0, x_j=1 => edges (i1->i0), (j0->j1)
    #
    # We only add edges when v[i]>v[j].  If v[i]<=v[j], no constraint.
    #----------------------------------------------------------------
    for i in range(Q):
        for j in range(i+1, Q):
            if v[i] > v[j]:
                # Then if p[i]==p[j], impossible
                if p[i] == p[j]:
                    print(0)
                    return
                elif p[i] < p[j]:
                    # Force x_i=1, x_j=0
                    adjacency[i0(i)].append(i1(i))  # x_i=0 => x_i=1 is impossible
                    adjacency[i1(j)].append(i0(j))  # x_j=1 => x_j=0 is impossible
                else:
                    # p[i] > p[j]
                    # Force x_i=0, x_j=1
                    adjacency[i1(i)].append(i0(i))
                    adjacency[i0(j)].append(i1(j))
            elif v[j] > v[i]:
                # Then v[j]>v[i] => i<j => no constraint from i to j
                # but from j's perspective j< i doesn't hold, so skip
                pass
            # if v[i]==v[j], also no constraint

    #----------------------------------------------------------------
    # 4) Find strongly connected components (SCC) using Tarjan or Kosaraju.
    #    We'll do Tarjan here for convenience.
    #----------------------------------------------------------------
    sys.setrecursionlimit(10**7)

    g = adjacency
    n = 2*Q

    stack = []
    in_stack = [False]*n
    scc_id = [-1]*n
    low = [0]*n
    dfn = [-1]*n
    current_index = 0
    scc_count = 0

    def tarjan(u):
        nonlocal current_index, scc_count
        dfn[u] = current_index
        low[u] = current_index
        current_index += 1
        stack.append(u)
        in_stack[u] = True

        for w in g[u]:
            if dfn[w] == -1:
                tarjan(w)
                low[u] = min(low[u], low[w])
            elif in_stack[w]:
                low[u] = min(low[u], dfn[w])

        if low[u] == dfn[u]:
            # root of SCC
            comp_id = scc_count
            scc_count += 1
            while True:
                x = stack.pop()
                in_stack[x] = False
                scc_id[x] = comp_id
                if x == u:
                    break

    for i in range(n):
        if dfn[i] == -1:
            tarjan(i)

    #----------------------------------------------------------------
    # 5) If for any i, i0 and i1 are in the same component => no solution.
    #----------------------------------------------------------------
    for i in range(Q):
        if scc_id[i0(i)] == scc_id[i1(i)]:
            print(0)
            return

    #----------------------------------------------------------------
    # 6) Build condensation graph of size scc_count.  Then topological-sort it.
    #----------------------------------------------------------------
    cond_graph = [[] for _ in range(scc_count)]
    in_deg = [0]*scc_count
    for u in range(n):
        c1 = scc_id[u]
        for w in g[u]:
            c2 = scc_id[w]
            if c1 != c2:
                cond_graph[c1].append(c2)
                in_deg[c2] += 1

    # Topological sort of condensation graph
    from collections import deque
    queue = deque()
    for c in range(scc_count):
        if in_deg[c] == 0:
            queue.append(c)
    topo_list = []
    while queue:
        cc = queue.popleft()
        topo_list.append(cc)
        for nxt in cond_graph[cc]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    # Now topo_list is a topological order (not reversed).
    # We'll want to process in reverse topological order for DP.
    # Just reverse it for convenience:
    topo_list.reverse()

    #----------------------------------------------------------------
    # 7) For transitive-closure in the DAG: dp[c] = bitmask of all comps
    #    reachable from c (including c).  We'll store each dp[c] as a
    #    Python int with scc_count bits.
    #    We'll do a standard "reverse topological" pass:
    #       dp[c] = (1 << c)
    #       for child in cond_graph[c]: dp[c] |= dp[child]
    #----------------------------------------------------------------
    dp_reach = [0]*scc_count
    for c in topo_list:
        # set its own bit
        dp_reach[c] = 1 << c
        for nxt in cond_graph[c]:
            dp_reach[c] |= dp_reach[nxt]

    #----------------------------------------------------------------
    # 8) Determine for each variable i if it is forced or free.
    #
    #    Let c0 = scc_id[i0], c1 = scc_id[i1].
    #    - If c1 in dp_reach[c0], then there's a path c0-> c1 => forced x_i=1.
    #    - If c0 in dp_reach[c1], then there's a path c1-> c0 => forced x_i=0.
    #    If both are true => no solutions.
    #    If neither is true => x_i is free => 2 ways.
    #----------------------------------------------------------------
    answer = 1
    for i in range(Q):
        c0 = scc_id[i0(i)]
        c1 = scc_id[i1(i)]
        # check if c1 is in dp_reach[c0]
        forced_one =  False
        if (dp_reach[c0] >> c1) & 1 == 1:
            forced_one = True
        forced_zero = False
        if (dp_reach[c1] >> c0) & 1 == 1:
            forced_zero = True

        if forced_one and forced_zero:
            # Contradiction
            print(0)
            return
        elif not forced_one and not forced_zero:
            # free
            answer = (answer * 2) % mod
        # else exactly one is forced => only 1 way for that variable

    print(answer % mod)


# Don't forget to call main!
if __name__ == "__main__":
    main()