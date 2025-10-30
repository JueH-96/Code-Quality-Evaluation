import sys
import heapq

def main():
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    A = [0]*(N-1)
    B = [0]*(N-1)
    X = [0]*(N-1)
    idx = 1
    for i in range(N-1):
        A[i] = int(input_data[idx]); B[i] = int(input_data[idx+1]); X[i] = int(input_data[idx+2])
        idx += 3
    
    # Build adjacency list
    # For each stage i (1 <= i < N), we have:
    # i -> i+1 (cost A[i])
    # i -> X[i] (cost B[i])
    adjacency = [[] for _ in range(N+1)]
    for i in range(1, N):
        adjacency[i].append((i+1, A[i-1]))
        adjacency[i].append((X[i-1], B[i-1]))
    
    # Dijkstra's algorithm
    INF = float('inf')
    dist = [INF]*(N+1)
    dist[1] = 0
    pq = [(0, 1)]  # (distance, node)
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        if dist[u] < current_dist:
            continue
        if u == N:
            print(current_dist)
            return
        for v, cost in adjacency[u]:
            new_dist = current_dist + cost
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    # If somehow not reachable (though problem statement implies it should be reachable),
    # we print dist[N] anyway.
    print(dist[N])

if __name__ == "__main__":
    main()