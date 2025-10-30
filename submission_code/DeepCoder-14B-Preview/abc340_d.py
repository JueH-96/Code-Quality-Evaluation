import heapq

def main():
    n = int(input())
    adj = [[] for _ in range(n + 1)]
    for i in range(1, n):
        a, b, x = map(int, input().split())
        adj[i].append((i + 1, a))
        adj[i].append((x, b))
    
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if u == n:
            break
        if current_dist > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    
    print(dist[n])

if __name__ == '__main__':
    main()