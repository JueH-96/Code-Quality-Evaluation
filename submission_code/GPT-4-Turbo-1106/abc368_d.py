from collections import defaultdict, deque

def bfs(graph, start, targets):
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    max_depth = 0
    total_visited = 0

    while queue:
        node, depth = queue.popleft()
        if node in targets:
            total_visited += 1
            max_depth = max(max_depth, depth)
        if total_visited == len(targets):
            break
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return max_depth

def main():
    N, K = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(N - 1):
        A, B = map(int, input().split())
        graph[A].append(B)
        graph[B].append(A)
    targets = set(map(int, input().split()))

    # Find the vertex that is closest to all targets
    min_distance = float('inf')
    best_vertex = None
    for vertex in targets:
        distance = bfs(graph, vertex, targets)
        if distance < min_distance:
            min_distance = distance
            best_vertex = vertex

    # The minimum number of vertices in the tree is the number of targets plus the max depth
    # from the best vertex to all other targets
    max_depth = bfs(graph, best_vertex, targets)
    result = K + max_depth
    print(result)

if __name__ == "__main__":
    main()