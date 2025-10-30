import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx]); idx +=1
    m = int(input[idx]); idx +=1
    k = int(input[idx]); idx +=1
    
    edges = []
    for _ in range(m):
        u = int(input[idx])-1; idx +=1
        v = int(input[idx])-1; idx +=1
        w = int(input[idx]); idx +=1
        edges.append((u, v, w))
    
    min_cost = float('inf')
    
    for candidate in itertools.combinations(edges, n-1):
        parent = list(range(n))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                parent[v_root] = u_root
        
        for u, v, _ in candidate:
            union(u, v)
        
        root = find(0)
        connected = True
        for node in range(1, n):
            if find(node) != root:
                connected = False
                break
        if connected:
            total = sum(w for _, _, w in candidate)
            mod_total = total % k
            if mod_total < min_cost:
                min_cost = mod_total
    
    print(min_cost % k)

if __name__ == "__main__":
    main()