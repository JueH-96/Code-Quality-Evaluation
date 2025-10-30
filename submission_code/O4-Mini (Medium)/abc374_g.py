def main():
    import sys
    sys.setrecursionlimit(10000)
    input = sys.stdin.readline

    N = int(input().strip())
    edges = []
    nodes_set = set()
    for _ in range(N):
        s = input().strip()
        u = ord(s[0]) - 65
        v = ord(s[1]) - 65
        edges.append((u, v))
        nodes_set.add(u)
        nodes_set.add(v)

    # Build adjacency for Tarjan
    g = {u: [] for u in nodes_set}
    for u, v in edges:
        g[u].append(v)

    # Tarjan's SCC
    index = {}
    lowlink = {}
    onstack = {}
    stack = []
    idx = 0
    comp_id = {}
    comp_count = 0

    def strongconnect(u):
        nonlocal idx, comp_count
        index[u] = idx
        lowlink[u] = idx
        idx += 1
        stack.append(u)
        onstack[u] = True

        for v in g[u]:
            if v not in index:
                strongconnect(v)
                lowlink[u] = min(lowlink[u], lowlink[v])
            elif onstack.get(v, False):
                lowlink[u] = min(lowlink[u], index[v])

        # u is root of SCC
        if lowlink[u] == index[u]:
            # pop stack until u
            while True:
                w = stack.pop()
                onstack[w] = False
                comp_id[w] = comp_count
                if w == u:
                    break
            comp_count += 1

    for u in nodes_set:
        if u not in index:
            strongconnect(u)

    # Build condensation graph
    K = comp_count
    cond_adj = [set() for _ in range(K)]
    indegree = [0] * K
    outdegree = [0] * K

    for u, v in edges:
        cu = comp_id[u]
        cv = comp_id[v]
        if cu != cv:
            if cv not in cond_adj[cu]:
                cond_adj[cu].add(cv)
                outdegree[cu] += 1
                indegree[cv] += 1

    # Build undirected adjacency for condensation components
    undirected = [set() for _ in range(K)]
    for cu in range(K):
        for cv in cond_adj[cu]:
            undirected[cu].add(cv)
            undirected[cv].add(cu)

    visited = [False] * K
    ans = 0

    from collections import deque

    for c in range(K):
        if visited[c]:
            continue
        # BFS/DFS this undirected CC
        dq = deque([c])
        visited[c] = True
        cc = []
        while dq:
            x = dq.popleft()
            cc.append(x)
            for y in undirected[x]:
                if not visited[y]:
                    visited[y] = True
                    dq.append(y)
        # For this CC, count sources and sinks in the DAG
        s = 0
        t = 0
        for x in cc:
            if indegree[x] == 0:
                s += 1
            if outdegree[x] == 0:
                t += 1
        # Minimum trails for this CC
        # At least one if edges exist; s and t are >=1 for nonempty cc
        ans += max(s, t)

    print(ans)

if __name__ == "__main__":
    main()