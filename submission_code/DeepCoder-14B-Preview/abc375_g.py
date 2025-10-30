import heapq

def dijkstra(start, n, adj):
    INF = float('inf')
    d = [INF] * (n + 1)
    d[start] = 0
    heap = [(0, start)]
    while heap:
        dist_u, u = heapq.heappop(heap)
        if dist_u > d[u]:
            continue
        for v, w in adj[u]:
            if d[v] > d[u] + w:
                d[v] = d[u] + w
                heapq.heappush(heap, (d[v], v))
    return d

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    roads = []
    adj = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        roads.append((A, B, C))
        adj[A].append((B, C))
        adj[B].append((A, C))
    
    d1 = dijkstra(1, N, adj)
    dN = dijkstra(N, N, adj)
    
    D = d1[N]
    
    S = []
    for i in range(M):
        A, B, C = roads[i]
        if (d1[A] + C + dN[B] == D) or (d1[B] + C + dN[A] == D):
            S.append(i)
    
    S_set = set(S)
    
    for i in range(M):
        if i in S_set:
            if len(S) > 1:
                print("No")
            else:
                print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()