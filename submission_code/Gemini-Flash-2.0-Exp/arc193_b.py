def solve():
    n = int(input())
    s = input()
    
    adj = [[] for _ in range(n + 1)]
    for i in range(n):
        adj[i].append((i + 1) % n)
        adj[(i + 1) % n].append(i)
        if s[i] == '1':
            adj[i].append(n)
            adj[n].append(i)
    
    edges = []
    for u in range(n + 1):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    num_edges = len(edges)
    
    distinct_degrees = set()
    
    for i in range(2**num_edges):
        degrees = [0] * (n + 1)
        
        for j in range(num_edges):
            if (i >> j) & 1:
                u, v = edges[j]
                degrees[v] += 1
            else:
                u, v = edges[j]
                degrees[u] += 1
        
        distinct_degrees.add(tuple(degrees))
    
    print(len(distinct_degrees) % 998244353)

solve()