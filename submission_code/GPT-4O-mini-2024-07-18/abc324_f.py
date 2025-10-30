import sys
import heapq

def max_beauty_cost_ratio(N, M, edges):
    # Create a graph representation
    graph = [[] for _ in range(N + 1)]
    for u, v, b, c in edges:
        graph[u].append((v, b, c))

    # Use a priority queue to maximize the ratio of beauty to cost
    # We will use a modified Dijkstra's algorithm
    max_ratio = [0.0] * (N + 1)
    max_ratio[1] = 0.0  # Starting point, no cost or beauty

    # Priority queue will store tuples of the form (-ratio, current_node, total_beauty, total_cost)
    pq = [(-0.0, 1, 0, 0)]  # Start from node 1 with 0 beauty and 0 cost

    while pq:
        neg_ratio, node, total_beauty, total_cost = heapq.heappop(pq)
        current_ratio = -neg_ratio

        # If we reach node N, we can check if we have a better ratio
        if node == N:
            return current_ratio

        # Explore neighbors
        for neighbor, beauty, cost in graph[node]:
            new_beauty = total_beauty + beauty
            new_cost = total_cost + cost
            if new_cost > 0:  # Avoid division by zero
                new_ratio = new_beauty / new_cost
                if new_ratio > max_ratio[neighbor]:
                    max_ratio[neighbor] = new_ratio
                    heapq.heappush(pq, (-new_ratio, neighbor, new_beauty, new_cost))

    return 0.0  # Fallback, should not reach here due to problem constraints

def main():
    input = sys.stdin.read
    data = input().splitlines()
    N, M = map(int, data[0].split())
    edges = []
    for i in range(1, M + 1):
        u, v, b, c = map(int, data[i].split())
        edges.append((u, v, b, c))
    
    result = max_beauty_cost_ratio(N, M, edges)
    print(f"{result:.10f}")

if __name__ == "__main__":
    main()