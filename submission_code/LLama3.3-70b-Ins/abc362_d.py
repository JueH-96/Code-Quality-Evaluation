import sys
import heapq

def dijkstra(graph, start):
    """
    Dijkstra's algorithm to find the shortest path from the start node to all other nodes.

    Args:
    graph: A dictionary representing the graph, where each key is a node and its value is another dictionary.
           The inner dictionary's keys are the neighboring nodes and its values are the edge weights.
    start: The node to start the search from.

    Returns:
    A dictionary where the keys are the nodes and the values are the shortest distances from the start node.
    """
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # Nodes can get added to the priority queue multiple times. We only
        # process a node the first time we remove it from the priority queue.
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # Only consider this new path if it is shorter than any path seen before.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

def main():
    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    edges = []
    for _ in range(M):
        u, v, b = map(int, input().split())
        edges.append((u-1, v-1, b))

    # Build graph
    graph = {i: {} for i in range(N)}
    for u, v, b in edges:
        graph[u][v] = b + A[v]
        graph[v][u] = b + A[u]

    # Find shortest distances from node 0 to all other nodes
    distances = dijkstra(graph, 0)

    # Print results
    for i in range(1, N):
        print(distances[i] + A[0], end=' ')

if __name__ == "__main__":
    main()