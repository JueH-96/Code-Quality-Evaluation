import heapq
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    M = int(data[idx])
    idx +=1
    
    edges = []
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        A = int(data[idx])
        idx +=1
        B = int(data[idx])
        idx +=1
        C = int(data[idx])
        idx +=1
        edges.append( (A, B, C) )
        adj[A].append( (B, C) )
        adj[B].append( (A, C) )
    
    def dijkstra(start):
        INF = float('inf')
        dist = [INF] * (N + 1)
        dist[start] = 0
        heap = []
        heapq.heappush(heap, (0, start))
        while heap:
            current_dist, u = heapq.heappop(heap)
            if current_dist > dist[u]:
                continue
            for v, c in adj[u]:
                if dist[v] > current_dist + c:
                    dist[v] = current_dist + c
                    heapq.heappush(heap, (dist[v], v))
        return dist
    
    d1 = dijkstra(1)
    dn = dijkstra(N)
    d = d1[N]
    
    dag_edges = []
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    
    for A, B, C in edges:
        u, v = A, B
        found = False
        if d1[u] + C + dn[v] == d:
            dag_edges.append( (u, v) )
            out_degree[u] += 1
            in_degree[v] += 1
            found = True
        elif d1[v] + C + dn[u] == d:
            dag_edges.append( (v, u) )
            out_degree[v] += 1
            in_degree[u] += 1
            found = True
        if not found:
            dag_edges.append(None)
    
    results = []
    for i in range(M):
        edge_info = dag_edges[i]
        if edge_info is None:
            results.append("No")
        else:
            u_dir, v_dir = edge_info
            cond1 = in_degree[v_dir] == 1
            cond2 = (out_degree[u_dir] == 1) and (u_dir != N)
            if cond1 or cond2:
                results.append("Yes")
            else:
                results.append("No")
    
    print('
'.join(results))

if __name__ == '__main__':
    main()