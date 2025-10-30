import sys
from collections import deque

def bfs(graph, start):
    """Perform BFS traversal from the given start node."""
    visited = [False] * len(graph)
    queue = deque([start])
    visited[start] = True
    distances = [0] * len(graph)
    while queue:
        node = queue.popleft()
        for neighbor, weight in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                distances[neighbor] = distances[node] + weight
                queue.append(neighbor)
    return distances

def solve():
    """Solve the problem."""
    n = int(input())
    graph = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v, w = map(int, input().split())
        u -= 1
        v -= 1
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Calculate distances from node 0 to all other nodes
    distances = bfs(graph, 0)

    # Sort distances in descending order
    sorted_distances = sorted(distances, reverse=True)

    # Calculate the answer for each K
    answers = []
    for k in range(1, n + 1):
        answer = 0
        for i in range(k):
            answer += sorted_distances[i]
        answer *= 2
        answers.append(answer)

    # Print the answers
    for answer in answers:
        print(answer)

if __name__ == "__main__":
    solve()