import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    # Disjoint Set Union (Union-Find)
    parent = list(range(n+1))
    rank = [0]*(n+1)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        else:
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
    
    idx = 2
    for _ in range(m):
        u = int(data[idx]); v = int(data[idx+1])
        idx += 2
        union(u, v)
    
    # Count number of distinct connected components
    roots = set()
    for i in range(1, n+1):
        roots.add(find(i))
    c = len(roots)
    
    # In a forest with c components on n vertices, the max edges is n-c.
    # We have m edges, so need to delete m - (n-c).
    result = m - (n - c)
    print(result)

if __name__ == "__main__":
    main()