import heapq

def dijkstra(graph, start, vertex_weights):
    N = len(graph)
    visited = [False] * N
    min_path_weights = [float('inf')] * N
    min_path_weights[start] = vertex_weights[start]
    priority_queue = [(vertex_weights[start], start)]

    while priority_queue:
        current_weight, current_vertex = heapq.heappop(priority_queue)
        if visited[current_vertex]:
            continue
        visited[current_vertex] = True

        for neighbor, edge_weight in graph[current_vertex]:
            path_weight = current_weight + edge_weight + vertex_weights[neighbor]
            if path_weight < min_path_weights[neighbor]:
                min_path_weights[neighbor] = path_weight
                heapq.heappush(priority_queue, (path_weight, neighbor))

    return min_path_weights

def main():
    N, M = map(int, input().split())
    vertex_weights = list(map(int, input().split()))
    graph = [[] for _ in range(N)]

    for _ in range(M):
        u, v, b = map(int, input().split())
        graph[u - 1].append((v - 1, b))
        graph[v - 1].append((u - 1, b))

    min_weights = dijkstra(graph, 0, vertex_weights)
    print(' '.join(map(str, min_weights[1:])))

if __name__ == "__main__":
    main()