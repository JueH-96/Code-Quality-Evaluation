from collections import deque
import sys

def bfs(graph, start, end):
    """Perform BFS to find the shortest path between start and end."""
    queue = deque([(start, 0)])
    visited = set()
    while queue:
        node, distance = queue.popleft()
        if node == end:
            return distance
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            queue.append((neighbor, distance + 1))
    return float('inf')

def main():
    """Read input, build graph, and find the maximum possible d."""
    N1, N2, M = map(int, sys.stdin.readline().split())
    N = N1 + N2
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        graph[a].append(b)
        graph[b].append(a)

    # Add edges between all nodes in the first group and all nodes in the second group
    for i in range(1, N1 + 1):
        for j in range(N1 + 1, N + 1):
            graph[i].append(j)
            graph[j].append(i)

    max_d = 0
    for u in range(1, N1 + 1):
        for v in range(N1 + 1, N + 1):
            # Temporarily remove the edge between u and v
            graph[u].remove(v)
            graph[v].remove(u)
            d = bfs(graph, 1, N)
            max_d = max(max_d, d)
            # Add the edge back
            graph[u].append(v)
            graph[v].append(u)

    print(max_d)

if __name__ == "__main__":
    main()