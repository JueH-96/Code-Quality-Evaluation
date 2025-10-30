from collections import deque
import sys

def bfs(graph, start, max_distance):
    """Perform BFS traversal from the start vertex up to max_distance."""
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)

    while queue:
        vertex, distance = queue.popleft()
        if distance > max_distance:
            break

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return visited

def main():
    """Read input, perform BFS for each guard, and print guarded vertices."""
    input_lines = sys.stdin.readlines()
    input_index = 0

    # Read the number of vertices, edges, and guards
    N, M, K = map(int, input_lines[input_index].split())
    input_index += 1

    # Create an adjacency list representation of the graph
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        a, b = map(int, input_lines[input_index].split())
        graph[a].append(b)
        graph[b].append(a)
        input_index += 1

    # Read guard positions and stamina
    guards = []
    for _ in range(K):
        p, h = map(int, input_lines[input_index].split())
        guards.append((p, h))
        input_index += 1

    # Perform BFS for each guard and mark guarded vertices
    guarded_vertices = set()
    for p, h in guards:
        guarded_vertices.update(bfs(graph, p, h))

    # Print the number of guarded vertices and their IDs
    print(len(guarded_vertices))
    print(' '.join(map(str, sorted(guarded_vertices))))

if __name__ == "__main__":
    main()