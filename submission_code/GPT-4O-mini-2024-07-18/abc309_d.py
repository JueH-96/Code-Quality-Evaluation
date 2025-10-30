def find_maximum_distance(N1, N2, M, edges):
    from collections import defaultdict, deque

    # Create adjacency list for the graph
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    # BFS to find the farthest node from vertex 1 in the first component
    def bfs(start):
        visited = [False] * (N1 + N2 + 1)
        queue = deque([start])
        visited[start] = True
        distance = {start: 0}
        farthest_node = start

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    distance[neighbor] = distance[node] + 1
                    if distance[neighbor] > distance[farthest_node]:
                        farthest_node = neighbor

        return farthest_node, distance

    # Find the farthest node from vertex 1 in the first component
    farthest_from_1, dist_from_1 = bfs(1)

    # Find the farthest node from the farthest node found above
    farthest_from_farthest, dist_from_farthest = bfs(farthest_from_1)

    # The diameter of the first component
    diameter_first_component = dist_from_farthest[farthest_from_farthest]

    # BFS from the last vertex N1 + N2 to find distances in the second component
    farthest_from_N1_plus_N2, dist_from_N1_plus_N2 = bfs(N1 + N2)

    # The maximum distance we can achieve by adding an edge
    max_distance = diameter_first_component + 1 + dist_from_N1_plus_N2[farthest_from_1]

    return max_distance

import sys
input = sys.stdin.read
data = input().splitlines()

N1, N2, M = map(int, data[0].split())
edges = [tuple(map(int, line.split())) for line in data[1:M + 1]]

result = find_maximum_distance(N1, N2, M, edges)
print(result)