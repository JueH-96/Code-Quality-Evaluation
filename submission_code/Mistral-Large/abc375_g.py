import sys
import heapq

def dijkstra(n, graph, start):
    distances = [float('inf')] * (n + 1)
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    edges = []
    graph = [[] for _ in range(N + 1)]

    for i in range(M):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        C = int(data[index])
        index += 1
        edges.append((A, B, C))
        graph[A].append((B, C))
        graph[B].append((A, C))

    original_distances = dijkstra(N, graph, 1)
    original_distance_to_N = original_distances[N]

    results = []

    for i in range(M):
        A, B, C = edges[i]
        graph[A].remove((B, C))
        graph[B].remove((A, C))

        new_distances = dijkstra(N, graph, 1)
        new_distance_to_N = new_distances[N]

        if new_distance_to_N != original_distance_to_N:
            results.append("Yes")
        else:
            results.append("No")

        graph[A].append((B, C))
        graph[B].append((A, C))

    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()