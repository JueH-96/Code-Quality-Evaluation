# YOUR CODE HERE
import itertools

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, t = map(int, input().split())
        edges.append(((u - 1, v - 1), t))

    q = int(input())
    for _ in range(q):
        line = list(map(int, input().split()))
        k = line[0]
        bridges = [x - 1 for x in line[1:]]

        min_time = float('inf')

        for perm in itertools.permutations(bridges):
            for directions in itertools.product([0, 1], repeat=k):
                current_time = 0
                current_node = 0
                
                for i in range(k):
                    bridge_index = perm[i]
                    edge = edges[bridge_index]
                    
                    if directions[i] == 0:
                        u, v = edge[0]
                        t = edge[1]
                    else:
                        v, u = edge[0]
                        t = edge[1]
                        
                    
                    dist = [[float('inf')] * n for _ in range(n)]
                    for i in range(n):
                        dist[i][i] = 0
                    for edge_info in edges:
                        u_edge, v_edge = edge_info[0]
                        t_edge = edge_info[1]
                        dist[u_edge][v_edge] = min(dist[u_edge][v_edge], t_edge)
                        dist[v_edge][u_edge] = min(dist[v_edge][u_edge], t_edge)

                    for kk in range(n):
                        for i in range(n):
                            for j in range(n):
                                dist[i][j] = min(dist[i][j], dist[i][kk] + dist[kk][j])
                    
                    current_time += t
                    current_node = v
                    
                dist = [[float('inf')] * n for _ in range(n)]
                for i in range(n):
                    dist[i][i] = 0
                for edge_info in edges:
                    u_edge, v_edge = edge_info[0]
                    t_edge = edge_info[1]
                    dist[u_edge][v_edge] = min(dist[u_edge][v_edge], t_edge)
                    dist[v_edge][u_edge] = min(dist[v_edge][u_edge], t_edge)

                for kk in range(n):
                    for i in range(n):
                        for j in range(n):
                            dist[i][j] = min(dist[i][j], dist[i][kk] + dist[kk][j])
                current_time += dist[current_node][n-1]
                min_time = min(min_time, current_time)

        print(min_time)

solve()