import heapq

def main():
    import sys
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
        B = int(input[ptr])
        ptr += 1
        adj[u].append((v, B + A[v-1]))
        adj[v].append((u, B + A[u-1]))
    
    INF = float('inf')
    dist = [INF] * (N + 1)
    dist[1] = A[0]
    heap = []
    heapq.heappush(heap, (dist[1], 1))
    
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    
    print(' '.join(map(str, dist[2:N+1])))

if __name__ == '__main__':
    main()