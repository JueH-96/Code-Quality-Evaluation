def solve():
    n, q = map(int, input().split())
    degrees = [0] * (n + 1)
    adj = [set() for _ in range(n + 1)]
    zero_degree_vertices = set(range(1, n + 1))
    
    for _ in range(q):
        query_parts = list(map(int, input().split()))
        query_type = query_parts[0]
        
        if query_type == 1:
            u, v = query_parts[1], query_parts[2]
            if u in zero_degree_vertices:
                zero_degree_vertices.remove(u)
            if v in zero_degree_vertices:
                zero_degree_vertices.remove(v)
            degrees[u] += 1
            degrees[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        elif query_type == 2:
            v = query_parts[1]
            neighbors_v = list(adj[v])
            for u in neighbors_v:
                degrees[u] -= 1
                if degrees[u] == 0:
                    zero_degree_vertices.add(u)
                adj[u].remove(v)
            
            initial_degree_v = degrees[v]
            adj[v].clear()
            degrees[v] = 0
            zero_degree_vertices.add(v)
            
        print(len(zero_degree_vertices))

if __name__ == '__main__':
    solve()