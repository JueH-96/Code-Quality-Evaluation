import sys
import threading

def main():
    import sys
    from itertools import combinations

    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    K = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        edges.append((u, v, w))

    best = K  # best cost found, in [0..K-1]
    # We'll enumerate all subsets of m edges of size n-1
    # Check if they form a spanning tree (no cycle, connected)
    # If so compute sum of weights mod K and track minimum.

    # Quick exit if K == 1 => all weights %1 == 0 => answer is 0
    if K == 1:
        print(0)
        return

    # DSU functions
    def find(par, x):
        # path compression
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    # Enumerate combinations
    # If best == 0, can't improve, break early
    edge_indices = range(m)
    for comb in combinations(edge_indices, n - 1):
        if best == 0:
            break
        # init DSU
        par = list(range(n))
        rank = [0] * n
        total = 0
        valid = True
        # try to union; if we find a cycle, drop this set
        for ei in comb:
            u, v, w = edges[ei]
            ru = find(par, u)
            rv = find(par, v)
            if ru == rv:
                valid = False
                break
            # union by rank
            if rank[ru] < rank[rv]:
                par[ru] = rv
            elif rank[ru] > rank[rv]:
                par[rv] = ru
            else:
                par[rv] = ru
                rank[ru] += 1
            total += w
        if not valid:
            continue
        # it's a tree (n-1 edges, no cycle) => connected
        cost = total % K
        if cost < best:
            best = cost

    print(best)

if __name__ == "__main__":
    main()