def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    edges = []
    for i in range(Q):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        new_node = N + i + 1
        edges.append((C, new_node, L))
        edges.append((C, new_node, R))
    
    edges.sort()
    
    parent = list(range(N + Q + 2))  # Nodes are 1-based up to N+Q
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    total_cost = 0
    components = N + Q  # Initial number of components
    
    for C, u, v in edges:
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            total_cost += C
            components -= 1
    
    if components == 1:
        print(total_cost)
    else:
        print(-1)

if __name__ == '__main__':
    main()