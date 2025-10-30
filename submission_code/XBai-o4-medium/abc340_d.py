import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N + 1)]  # 1-based to N
    
    for i in range(1, N):  # i from 1 to N-1
        A = int(input[ptr])
        B = int(input[ptr+1])
        X = int(input[ptr+2])
        ptr += 3
        adj[i].append((i + 1, A))
        adj[i].append((X, B))
    
    INF = float('inf')
    dist = [INF] * (N + 1)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    
    print(dist[N])

if __name__ == '__main__':
    main()