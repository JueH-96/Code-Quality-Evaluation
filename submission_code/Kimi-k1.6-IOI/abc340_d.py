import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    
    for i in range(1, N):
        A = int(data[idx])
        B = int(data[idx+1])
        X = int(data[idx+2])
        idx +=3
        adj[i].append((i+1, A))
        adj[i].append((X, B))
    
    INF = float('inf')
    dist = [INF] * (N +1)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        d, u = heapq.heappop(heap)
        if u == N:
            print(d)
            return
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    
    # The problem guarantees a path exists, so no need for further checks

if __name__ == "__main__":
    main()