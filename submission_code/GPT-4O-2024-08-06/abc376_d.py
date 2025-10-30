# YOUR CODE HERE
def find_min_cycle_with_vertex_1(n, m, edges):
    from collections import deque, defaultdict

    # Create adjacency list
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)

    # BFS to find the shortest cycle including vertex 1
    def bfs(start):
        queue = deque([(start, 0)])  # (current_node, distance_from_start)
        visited = {start: 0}  # node: distance_from_start
        min_cycle_length = float('inf')

        while queue:
            current, dist = queue.popleft()

            for neighbor in graph[current]:
                if neighbor == start:
                    # Found a cycle including vertex 1
                    min_cycle_length = min(min_cycle_length, dist + 1)
                elif neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return min_cycle_length if min_cycle_length != float('inf') else -1

    return bfs(1)

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
edges = [(int(data[i * 2 + 2]), int(data[i * 2 + 3])) for i in range(M)]

# Find and print the result
result = find_min_cycle_with_vertex_1(N, M, edges)
print(result)