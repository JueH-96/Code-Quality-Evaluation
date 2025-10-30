import sys
from collections import deque, defaultdict

def find_minimum_cycle(N, M, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)

    def bfs(start):
        queue = deque([(start, 0, -1)])  # (current_node, distance, previous_node)
        visited = [-1] * (N + 1)
        visited[start] = 0
        min_cycle_length = float('inf')

        while queue:
            current, dist, prev = queue.popleft()

            if current == start and dist > 1:
                min_cycle_length = min(min_cycle_length, dist)
                continue

            for neighbor in graph[current]:
                if visited[neighbor] == -1:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1, current))
                elif neighbor == start and prev != neighbor:
                    min_cycle_length = min(min_cycle_length, dist + 1)

        return min_cycle_length if min_cycle_length != float('inf') else -1

    return bfs(1)

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
edges = []
index = 2
for _ in range(M):
    a = int(data[index])
    b = int(data[index + 1])
    edges.append((a, b))
    index += 2

result = find_minimum_cycle(N, M, edges)
print(result)