def solve():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    edges_data = input_data[2+N:]
    
    # Build adjacency list
    # adjacency[v] = list of (neighbor, edge_weight)
    adjacency = [[] for _ in range(N)]
    idx = 0
    for _ in range(M):
        u = int(edges_data[idx]) - 1
        v = int(edges_data[idx+1]) - 1
        b = int(edges_data[idx+2])
        idx += 3
        adjacency[u].append((v, b))
        adjacency[v].append((u, b))
    
    # Dijkstra's algorithm
    INF = 10**18
    dist = [INF]*N
    dist[0] = A[0]  # cost to start: includes A_1 for vertex 1 (index 0)
    pq = [(dist[0], 0)]
    
    while pq:
        current_dist, node = heapq.heappop(pq)
        if current_dist > dist[node]:
            continue
        for neighbor, edge_weight in adjacency[node]:
            # cost from node to neighbor includes the edge weight plus the neighbor's vertex weight
            cost = dist[node] + edge_weight + A[neighbor]
            if cost < dist[neighbor]:
                dist[neighbor] = cost
                heapq.heappush(pq, (cost, neighbor))
    
    # Print answers for vertices 2..N (index 1..N-1)
    print(' '.join(map(str, dist[1:])))
    
def main():
    solve()

if __name__ == "__main__":
    main()