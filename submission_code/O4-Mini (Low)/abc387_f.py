import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    mod = 998244353
    data = sys.stdin.read().split()
    N = int(data[0]); M = int(data[1])
    A = [int(x)-1 for x in data[2:2+N]]
    # build graph and reverse
    g = [[] for _ in range(N)]
    gr = [[] for _ in range(N)]
    for i in range(N):
        v = A[i]
        g[i].append(v)
        gr[v].append(i)
    # Kosaraju
    used = [False]*N
    order = []
    def dfs(v):
        used[v] = True
        for w in g[v]:
            if not used[w]:
                dfs(w)
        order.append(v)
    for i in range(N):
        if not used[i]:
            dfs(i)
    comp = [-1]*N
    cid = 0
    def dfs2(v):
        comp[v] = cid
        for w in gr[v]:
            if comp[w]==-1:
                dfs2(w)
    for v in reversed(order):
        if comp[v]==-1:
            dfs2(v)
            cid += 1
    C = cid  # number of SCCs
    # build SCC DAG edges: u->v if any i in u points to A[i] in v and u!=v
    edges = [[] for _ in range(C)]
    indeg = [0]*C
    for i in range(N):
        u = comp[i]
        v = comp[A[i]]
        if u!=v:
            edges[u].append(v)
    # remove duplicates
    for u in range(C):
        edges[u] = list(set(edges[u]))
    # compute indegree in DAG on edges reversed for DP tree: but for DP we want for each v list of children u with edge u->v
    children = [[] for _ in range(C)]
    roots = [True]*C
    for u in range(C):
        for v in edges[u]:
            children[v].append(u)
            roots[u] = False
    # roots are those with no parent => they are tails? Actually those u with no outgoing edges in original condense DAG, i.e. u with edges to themselves? But here roots after reverse children: nodes with roots[u]==True are those with no incoming edges in original edges, i.e. no child->u, meaning no one points to them, so they are leaves in original pointer forest, but those are starting points.
    # Actually components independent are connected via edges. We want to identify each weak component's sink (cycle) as DP root: those with no outgoing edges in original edges[u]. These satisfy edges[u]==[]. Let's pick them.
    dp_roots = [u for u in range(C) if len(edges[u])==0]
    # Prepare topo order of SCC DAG so that children before parent: we want an order sorted by distance to dp_root; simpler: do DFS from each dp_root on children graph.
    visited = [False]*C
    topo = []
    def dfs3(u):
        visited[u] = True
        for c in children[u]:
            if not visited[c]:
                dfs3(c)
        topo.append(u)
    for r in dp_roots:
        if not visited[r]:
            dfs3(r)
    # topo now children->...->root
    # dp_sub[u][k] for k=1..M
    # to save memory, use list of lists
    dp_sub = [None]*C
    pref = [None]*C
    for u in topo:
        # compute dp_sub[u]
        # if leaf (children[u] empty): dp_sub[u][k] = 1 for all k
        if not children[u]:
            arr = [1]*(M+1)  # 1-indexed
        else:
            # for each k compute product of pref[child][k]
            arr = [0]*(M+1)
            for k in range(1, M+1):
                prod = 1
                for c in children[u]:
                    prod = prod * pref[c][k] % mod
                arr[k] = prod
        # prefix sums
        pr = [0]*(M+1)
        s = 0
        for k in range(1, M+1):
            s = (s + arr[k]) % mod
            pr[k] = s
        dp_sub[u] = arr
        pref[u] = pr
    # For each component (dp_root), its count = sum_{k=1..M} dp_sub[root][k]
    result = 1
    for r in dp_roots:
        total = pref[r][M]  # sum all
        result = result * total % mod
    print(result)

if __name__ == "__main__":
    main()