from heapq import heappush, heappop

def dijkstra(n, adj, adj2, start):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(pq, (dist[v], v))
        for v, w in adj2[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(pq, (dist[v], v))
    return dist

def main():
    n, a, b, c = map(int, input().split())
    adj = [[] for _ in range(n)]
    adj2 = [[] for _ in range(n)]
    for i in range(n):
        d = list(map(int, input().split()))
        for j in range(n):
            if i != j:
                adj[i].append((j, d[j] * a))
                adj2[i].append((j, d[j] * b + c))
    print(dijkstra(n, adj, adj2, 0)[n-1])

if __name__ == "__main__":
    main()