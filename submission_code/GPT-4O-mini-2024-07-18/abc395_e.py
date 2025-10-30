from collections import defaultdict, deque
import sys
import heapq

def minimum_cost_to_reach_n(N, M, X, edges):
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)

    # Build the graph and the reverse graph
    for u, v in edges:
        graph[u].append(v)
        reverse_graph[v].append(u)

    # Dijkstra's algorithm to find the minimum cost
    # Priority queue for (cost, vertex)
    pq = [(0, 1)]  # Start from vertex 1 with cost 0
    min_cost = {i: float('inf') for i in range(1, N + 1)}
    min_cost[1] = 0

    while pq:
        current_cost, current_vertex = heapq.heappop(pq)

        if current_cost > min_cost[current_vertex]:
            continue

        # Move along the edges
        for neighbor in graph[current_vertex]:
            new_cost = current_cost + 1
            if new_cost < min_cost[neighbor]:
                min_cost[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor))

        # Reverse the edges
        new_cost = current_cost + X
        for neighbor in reverse_graph[current_vertex]:
            if new_cost < min_cost[neighbor]:
                min_cost[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor))

    return min_cost[N]

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N, M, X
    N, M, X = map(int, data[0].split())
    
    # Read edges
    edges = []
    for i in range(1, M + 1):
        u, v = map(int, data[i].split())
        edges.append((u, v))
    
    # Calculate the minimum cost to reach vertex N
    result = minimum_cost_to_reach_n(N, M, X, edges)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()