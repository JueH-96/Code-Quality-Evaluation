import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx +=2
    A = list(map(int, data[idx:idx+N]))
    idx +=N
    A = [0] + A  # 1-based indexing
    
    edges = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(data[idx])
        v = int(data[idx+1])
        b = int(data[idx+2])
        idx +=3
        edges[u].append((v, b + A[v]))
        edges[v].append((u, b + A[u]))
    
    INF = float('inf')
    dist = [INF] * (N+1)
    dist[1] = A[1]
    heap = []
    heapq.heappush(heap, (dist[1], 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for v, cost in edges[u]:
            if dist[v] > current_dist + cost:
                dist[v] = current_dist + cost
                heapq.heappush(heap, (dist[v], v))
    
    print(' '.join(map(str, dist[2:N+1])))

if __name__ == "__main__":
    main()