import sys
from collections import defaultdict, deque

def bfs(start, target_vertices, graph):
    queue = deque([start])
    visited = set([start])
    while queue:
        node = queue.popleft()
        if node in target_vertices:
            target_vertices.remove(node)
            if not target_vertices:
                return visited
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])

    graph = defaultdict(list)
    for i in range(2, 2 + 2 * (N - 1), 2):
        A = int(data[i])
        B = int(data[i + 1])
        graph[A].append(B)
        graph[B].append(A)

    target_vertices = set(map(int, data[2 + 2 * (N - 1):]))

    # Start BFS from the first target vertex
    start_vertex = target_vertices.pop()
    target_vertices.add(start_vertex)

    result = bfs(start_vertex, target_vertices, graph)

    print(len(result))

if __name__ == "__main__":
    main()