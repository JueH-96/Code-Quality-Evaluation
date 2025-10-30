import sys
from collections import deque

def bfs(graph, start):
    """Perform BFS traversal from the given start node."""
    visited = [False] * (n + 1)
    distance = [0] * (n + 1)
    queue = deque([start])
    visited[start] = True

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return distance

def find_farthest_node(graph, start):
    """Find the farthest node from the given start node."""
    distance = bfs(graph, start)
    max_distance = max(distance)
    farthest_node = distance.index(max_distance)
    return farthest_node

def remove_node(graph, node):
    """Remove the given node from the graph."""
    for neighbor in graph[node]:
        graph[neighbor].remove(node)
    graph[node].clear()

def solve():
    """Solve the problem."""
    global n
    n = int(input())
    graph = [[] for _ in range(n + 1)]

    for _ in range(n - 1):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)

    result = []
    for _ in range(n // 2):
        # Find the farthest leaf node from node 1
        farthest_node = find_farthest_node(graph, 1)
        # Find the other leaf node
        other_leaf_node = None
        for node in range(1, n + 1):
            if len(graph[node]) == 1:
                other_leaf_node = node
                break

        # Remove the two leaf nodes
        remove_node(graph, farthest_node)
        remove_node(graph, other_leaf_node)

        # Add the pair to the result
        result.append((farthest_node, other_leaf_node))

    # Print the result
    for pair in result:
        print(*pair)

if __name__ == "__main__":
    solve()