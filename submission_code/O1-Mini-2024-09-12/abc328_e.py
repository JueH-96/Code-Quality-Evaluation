# YOUR CODE HERE
import sys
import itertools

def main():
    import sys
    from itertools import combinations

    sys.setrecursionlimit(1000000)
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    K = int(data[idx]); idx +=1
    edges = []
    for _ in range(M):
        u = int(data[idx])-1; idx +=1
        v = int(data[idx])-1; idx +=1
        w = int(data[idx]); idx +=1
        edges.append( (w, u, v) )
    min_cost = None
    for comb in itertools.combinations(edges, N-1):
        parent = list(range(N))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        connected = True
        for w, u, v in comb:
            pu = find(u)
            pv = find(v)
            if pu != pv:
                parent[pu] = pv
        roots = set(find(i) for i in range(N))
        if len(roots) == 1:
            total = sum(edge[0] for edge in comb)
            mod = total % K
            if min_cost is None or mod < min_cost:
                min_cost = mod
                if min_cost ==0:
                    break
    print(min_cost)

if __name__ == "__main__":
    main()