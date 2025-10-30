import sys
from collections import deque

def bfs(graph, start, n):
    """Perform BFS traversal from the start node and return the distance to all other nodes."""
    distance = [float('inf')] * (n + 1)
    distance[start] = 0
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if distance[neighbor] == float('inf'):
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return distance

def main():
    """Read input, build the graph, calculate f(x) for each node, and find the minimum."""
    n = int(input())
    graph = [[] for _ in range(n + 1)]

    # Build the graph
    for _ in range(n - 1):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)

    # Read the sequence C
    c = list(map(int, input().split()))

    # Initialize the minimum value of f(x)
    min_f = float('inf')

    # Calculate f(x) for each node
    for i in range(1, n + 1):
        distance = bfs(graph, i, n)
        f_x = sum(c[j - 1] * distance[j] for j in range(1, n + 1))
        min_f = min(min_f, f_x)

    print(min_f)

if __name__ == "__main__":
    main()