import sys
import heapq

def dijkstra(N, graph, weights):
    min_weight = [float('inf')] * (N + 1)
    min_weight[1] = weights[0]  # Weight of vertex 1
    priority_queue = [(weights[0], 1)]  # (current_weight, vertex)

    while priority_queue:
        current_weight, u = heapq.heappop(priority_queue)

        if current_weight > min_weight[u]:
            continue

        for v, edge_weight in graph[u]:
            new_weight = current_weight + edge_weight + weights[v - 1]
            if new_weight < min_weight[v]:
                min_weight[v] = new_weight
                heapq.heappush(priority_queue, (new_weight, v))

    return min_weight

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Read vertex weights
    weights = list(map(int, data[1].split()))
    
    # Read edges and build the graph
    graph = [[] for _ in range(N + 1)]
    for i in range(2, 2 + M):
        u, v, b = map(int, data[i].split())
        graph[u].append((v, b))
        graph[v].append((u, b))
    
    # Run Dijkstra's algorithm from vertex 1
    min_weights = dijkstra(N, graph, weights)
    
    # Collect results for vertices 2 to N
    results = [min_weights[i] for i in range(2, N + 1)]
    
    # Print results
    print(" ".join(map(str, results)))

if __name__ == "__main__":
    main()