def solve():
    n, q = map(int, input().split())
    adj = [set() for _ in range(n)]
    degree = [0] * n
    
    for _ in range(q):
        query_parts = list(map(int, input().split()))
        query_type = query_parts[0]
        
        if query_type == 1:
            u, v = query_parts[1], query_parts[2]
            adj[u-1].add(v)
            adj[v-1].add(u)
            degree[u-1] += 1
            degree[v-1] += 1
        elif query_type == 2:
            v = query_parts[1]
            neighbors_v = list(adj[v-1])
            for u in neighbors_v:
                if v in adj[u-1]:
                    adj[u-1].remove(v)
                    degree[u-1] -= 1
            adj[v-1].clear()
            degree[v-1] = 0
            
        degree_zero_count = 0
        for i in range(n):
            if degree[i] == 0:
                degree_zero_count += 1
        print(degree_zero_count)

if __name__ == '__main__':
    solve()