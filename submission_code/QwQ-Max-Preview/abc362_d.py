import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    M = int(data[idx])
    idx +=1
    
    A = list(map(int, data[idx:idx+N]))
    idx += N
    
    adj = [[] for _ in range(N+1)]  # 1-based
    
    for _ in range(M):
        u = int(data[idx])
        idx +=1
        v = int(data[idx])
        idx +=1
        b = int(data[idx])
        idx +=1
        
        # Add u -> v
        adj[u].append( (v, b + A[v-1]) )
        # Add v -> u
        adj[v].append( (u, b + A[u-1]) )
    
    # Dijkstra's algorithm
    INF = float('inf')
    dist = [INF] * (N + 1)
    dist[1] = A[0]  # node 1's weight is A[0]
    
    heap = []
    heapq.heappush(heap, (dist[1], 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for (v, cost) in adj[u]:
            if dist[v] > current_dist + cost:
                dist[v] = current_dist + cost
                heapq.heappush(heap, (dist[v], v))
    
    # Collect results for nodes 2 to N
    result = [str(dist[i]) for i in range(2, N+1)]
    print(' '.join(result))

if __name__ == '__main__':
    main()