import sys
import heapq

def min_travel_distance(N, edges):
    # Create a graph representation
    graph = [[] for _ in range(N+1)]
    for A, B, C in edges:
        graph[A].append((B, C))
        graph[B].append((A, C))

    # Priority queue for Dijkstra's algorithm
    pq = [(0, 1)]  # (distance, node)
    visited = [False] * (N + 1)
    total_distance = 0

    while pq:
        dist, node = heapq.heappop(pq)
        if visited[node]:
            continue
        visited[node] = True
        total_distance += dist
        for neighbor, weight in graph[node]:
            if not visited[neighbor]:
                heapq.heappush(pq, (weight, neighbor))

    return total_distance

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N-1):
        A = int(data[index])
        B = int(data[index + 1])
        C = int(data[index + 2])
        edges.append((A, B, C))
        index += 3

    result = min_travel_distance(N, edges)
    print(result)

if __name__ == "__main__":
    main()