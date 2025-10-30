from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    info = []
    for _ in range(M):
        A, B, X, Y = map(int, input().split())
        info.append((A, B, X, Y))

    # Create a graph where each node represents a person
    # and the edges represent the relative positions
    graph = defaultdict(list)
    for A, B, X, Y in info:
        graph[A].append((B, X, Y))
        graph[B].append((A, -X, -Y))

    # Perform DFS to find the coordinates of each person
    visited = set()
    coordinates = [None] * (N + 1)
    coordinates[1] = (0, 0)

    def dfs(node):
        visited.add(node)
        for neighbor, dx, dy in graph[node]:
            if neighbor not in visited:
                if coordinates[neighbor] is None:
                    coordinates[neighbor] = (coordinates[node][0] + dx, coordinates[node][1] + dy)
                    dfs(neighbor)
                else:
                    if (coordinates[neighbor][0] != coordinates[node][0] + dx or
                        coordinates[neighbor][1] != coordinates[node][1] + dy):
                        coordinates[0] = None
                        return

    for i in range(1, N + 1):
        if i not in visited:
            dfs(i)

    for i in range(1, N + 1):
        if coordinates[i] is None:
            print("undecidable")
        else:
            print(*coordinates[i])

solve()