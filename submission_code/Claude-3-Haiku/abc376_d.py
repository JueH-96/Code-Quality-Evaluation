from collections import defaultdict

def find_min_cycle_with_1(n, m, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)

    visited = [False] * (n + 1)
    parent = [None] * (n + 1)
    min_cycle_len = float('inf')

    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                parent[neighbor] = node
                if dfs(neighbor):
                    return True
            elif neighbor == 1 and parent[node] != neighbor:
                nonlocal min_cycle_len
                cycle_len = 1
                curr = node
                while curr != neighbor:
                    cycle_len += 1
                    curr = parent[curr]
                min_cycle_len = min(min_cycle_len, cycle_len)
                return True
        return False

    dfs(1)
    return min_cycle_len if min_cycle_len != float('inf') else -1

# Read input
n, m = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]

# Solve the problem
result = find_min_cycle_with_1(n, m, edges)
print(result)