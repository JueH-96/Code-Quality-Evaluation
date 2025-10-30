def main():
    import sys
    import heapq
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Adjacency list: for each city store (neighbor, length)
    adjacency = [[] for _ in range(N)]
    
    # Read edges
    idx = 1
    total_length = 0
    for _ in range(N-1):
        A = int(input_data[idx]) - 1
        B = int(input_data[idx+1]) - 1
        C = int(input_data[idx+2])
        idx += 3
        adjacency[A].append((B, C))
        adjacency[B].append((A, C))
        total_length += C
    
    # Dijkstra's algorithm to find distances from a start node
    def dijkstra(start):
        dist = [float('inf')] * N
        dist[start] = 0
        pq = [(0, start)]  # (distance, node)
        while pq:
            cd, cn = heapq.heappop(pq)
            if cd > dist[cn]:
                continue
            for (nn, cost) in adjacency[cn]:
                nd = cd + cost
                if nd < dist[nn]:
                    dist[nn] = nd
                    heapq.heappush(pq, (nd, nn))
        return dist
    
    # 1) Run Dijkstra from an arbitrary node (0) to find the farthest node
    dist_from_0 = dijkstra(0)
    farthest_node = max(range(N), key=lambda x: dist_from_0[x])
    
    # 2) Run Dijkstra from that farthest node to find the diameter
    dist_from_far = dijkstra(farthest_node)
    diameter = max(dist_from_far)
    
    # Formula: 2 * (sum of all edges) - diameter
    answer = 2 * total_length - diameter
    print(answer)

# Call main() to ensure the solution is executed
if __name__ == "__main__":
    main()