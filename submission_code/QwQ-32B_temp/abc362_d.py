import sys
import heapq

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    adj = [[] for _ in range(N + 1)]  # 1-based indexing

    for _ in range(M):
        u, v, b = map(int, sys.stdin.readline().split())
        # Add edge u -> v with weight B + A[v]
        adj[u].append((v, b + A[v - 1]))
        # Add edge v -> u with weight B + A[u]
        adj[v].append((u, b + A[u - 1]))
    
    INF = float('inf')
    dist = [INF] * (N + 1)
    dist[1] = A[0]  # Starting node's A is included

    heap = []
    heapq.heappush(heap, (dist[1], 1))

    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        for v, weight in adj[u]:
            if dist[v] > current_dist + weight:
                dist[v] = current_dist + weight
                heapq.heappush(heap, (dist[v], v))
    
    # Prepare the output for vertices 2..N
    output = []
    for i in range(2, N + 1):
        output.append(str(dist[i]))
    print(' '.join(output))

if __name__ == "__main__":
    main()