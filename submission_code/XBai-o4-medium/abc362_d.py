import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        # Add both directions with adjusted weights
        adj[u].append((v, b + A[v-1]))
        adj[v].append((u, b + A[u-1]))
    
    INF = 10**18
    dist = [INF] * (N + 1)
    dist[1] = A[0]
    heap = []
    heapq.heappush(heap, (dist[1], 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for (v, cost) in adj[u]:
            if dist[v] > dist[u] + cost:
                dist[v] = dist[u] + cost
                heapq.heappush(heap, (dist[v], v))
    
    print(' '.join(map(str, dist[2:N+1])))

if __name__ == "__main__":
    main()