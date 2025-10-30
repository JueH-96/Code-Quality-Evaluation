import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    A = list(map(int, data[idx:idx+N]))
    idx += N
    edges = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(data[idx])
        v = int(data[idx+1])
        b = int(data[idx+2])
        idx += 3
        # Since the graph is undirected, add both directions
        edges[u].append((v, b))
        edges[v].append((u, b))
    
    # Initialize distances
    dist = [float('inf')] * (N+1)
    dist[1] = A[0]  # Starting at vertex 1, its weight is A[0]
    
    # Priority queue: (distance, vertex)
    pq = []
    heapq.heappush(pq, (dist[1], 1))
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v, b in edges[u]:
            # The path weight is current_dist + A[v-1] + b
            new_dist = current_dist + A[v-1] + b
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    # Output the results for vertices 2 to N
    result = []
    for i in range(2, N+1):
        result.append(str(dist[i]))
    print(' '.join(result))

if __name__ == "__main__":
    main()