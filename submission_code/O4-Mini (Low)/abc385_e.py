import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u,v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    parent = [0]*(N+1)
    sz = [0]*(N+1)
    # dfs1 to compute subtree sizes rooted at 1
    def dfs(u,p):
        parent[u]=p
        sz[u]=1
        for v in adj[u]:
            if v==p: continue
            dfs(v,u)
            sz[u]+=sz[v]
    dfs(1,0)
    best_prod = 0
    # for each node c, consider as center
    # precompute for fast access: parent and sz
    for c in range(1, N+1):
        d = len(adj[c])
        if d == 0:
            continue
        m_list = []
        # for each neighbor h, compute m[h]=degree[h]-1
        for h in adj[c]:
            # count children of h when rooting at c is deg[h]-1
            m = len(adj[h]) - 1
            # also subtree size of h under c, but not needed here
            m_list.append(m)
        m_list.sort()
        # candidate y=0
        # y=0 gives prod = d*(0+1)=d
        best = d
        # for each unique m in m_list where m>=1
        # using sorted list, for j in range(d):
        for j, m in enumerate(m_list):
            if m <= 0:
                continue
            # number of eligible hubs k = count of m_list entries >= m = d - j
            k = d - j
            prod = k * (m + 1)
            if prod > best:
                best = prod
        if best > best_prod:
            best_prod = best
    # minimal deletions = total non-center nodes (N-1) minus best_prod
    ans = (N - 1) - best_prod
    print(ans)

if __name__ == "__main__":
    main()