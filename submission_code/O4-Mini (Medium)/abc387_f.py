import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    N = int(data[0]); M = int(data[1])
    A = list(map(int, data[2:] ))
    mod = 998244353

    # Build reversed adjacency: for each node, list of children u where A[u] = node+1
    rev_adj = [[] for _ in range(N)]
    for u in range(N):
        p = A[u] - 1
        rev_adj[p].append(u)

    # Find components and their cycles
    comp = [-1] * N
    cycles = []  # list of lists of cycle nodes per component
    comp_idx = 0

    for i in range(N):
        if comp[i] != -1:
            continue
        # Walk until we hit a repeated node in this walk
        u = i
        path = []
        idx = {}
        while True:
            if u in idx:
                # cycle found
                cyc = path[idx[u]:]
                cycles.append(cyc)
                break
            idx[u] = len(path)
            path.append(u)
            u = A[u] - 1
        # assign component index to all nodes in the walked path
        for node in path:
            comp[node] = comp_idx
        comp_idx += 1

    # Mark cycle nodes
    in_cycle = [False] * N
    for cyc in cycles:
        for u in cyc:
            in_cycle[u] = True

    ans = 1

    # DFS to compute g_u for subtree rooted at u (including u),
    # g_u[y] = number of ways to assign subtree if x_u = y+1.
    def dfs(u):
        # start with all 1's
        g = [1] * M
        # for each child in reversed tree (excluding cycle-nodes)
        for v in rev_adj[u]:
            if in_cycle[v]:
                continue
            gv = dfs(v)
            # build prefix sums of gv
            pref = [0] * M
            s = 0
            for i in range(M):
                s += gv[i]
                if s >= mod: s -= mod
                pref[i] = s
            # multiply g by pref entrywise
            for i in range(M):
                g[i] = (g[i] * pref[i]) % mod
        return g

    # Process each component
    for cyc in cycles:
        # comp_g[y] = product of g_root[y] over all cycle-nodes roots, for y=0..M-1
        comp_g = [1] * M
        for root in cyc:
            gr = dfs(root)
            # multiply into comp_g
            for i in range(M):
                comp_g[i] = (comp_g[i] * gr[i]) % mod
        # sum over y=0..M-1
        total = sum(comp_g) % mod
        ans = (ans * total) % mod

    print(ans)

if __name__ == "__main__":
    main()