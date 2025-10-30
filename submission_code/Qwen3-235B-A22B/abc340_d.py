import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N):
        A = int(input[ptr])
        B = int(input[ptr + 1])
        X = int(input[ptr + 2])
        ptr += 3
        adj[i].append((i + 1, A))
        adj[i].append((X, B))
    
    INF = 1 << 60
    dist = [INF] * (N + 1)
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
        for v, cost in adj[u]:
            if dist[v] > d + cost:
                dist[v] = d + cost
                heapq.heappush(heap, (dist[v], v))
    
    # Unreachable as per problem constraints
    print(-1)

if __name__ == "__main__":
    main()