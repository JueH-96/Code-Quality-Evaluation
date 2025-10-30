def solve():
    n, m, q = map(int, input().split())
    edges = []
    for _ in range(m):
        edges.append(list(map(int, input().split())))
    
    closed_edges = [False] * m
    
    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            closed_edges[query[1] - 1] = True
        else:
            x, y = query[1], query[2]
            
            dist = [[float('inf')] * n for _ in range(n)]
            for i in range(n):
                dist[i][i] = 0
            
            for i in range(m):
                if not closed_edges[i]:
                    u, v, c = edges[i]
                    dist[u - 1][v - 1] = c
                    dist[v - 1][u - 1] = c
            
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
            
            if dist[x - 1][y - 1] == float('inf'):
                print(-1)
            else:
                print(dist[x - 1][y - 1])

solve()