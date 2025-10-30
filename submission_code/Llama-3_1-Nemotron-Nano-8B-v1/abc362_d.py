import sys
import heapq

def main():
    input = sys.stdin.read().split()
    idx = 0
    N, M = int(input[idx]), int(input[idx+1])
    idx +=2
    A = list(map(int, input[idx:idx+N]))
    idx +=N
    A = [0] + A  # 1-based indexing
    
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        U = int(input[idx])
        V = int(input[idx+1])
        B = int(input[idx+2])
        idx +=3
        adj[U].append((V, B + A[V]))
        adj[V].append((U, B + A[U]))
    
    INF = float('inf')
    dist = [INF] * (N+1)
    dist[1] = A[1]
    heap = []
    heapq.heappush(heap, (A[1], 1))
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for v, cost in adj[u]:
            if dist[v] > dist[u] + cost:
                dist[v] = dist[u] + cost
                heapq.heappush(heap, (dist[v], v))
    
    output = []
    for i in range(2, N+1):
        output.append(str(dist[i]))
    print(' '.join(output))

if __name__ == '__main__':
    main()