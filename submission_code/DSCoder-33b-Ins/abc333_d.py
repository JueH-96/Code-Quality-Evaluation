from collections import defaultdict, deque

def solve():
    n = int(input())
    graph = defaultdict(list)
    degree = defaultdict(int)
    for _ in range(n-1):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
        degree[u] += 1
        degree[v] += 1

    leaves = deque([node for node in graph if degree[node] == 1])
    total_nodes = n
    operations = 0

    while total_nodes > 1:
        for _ in range(len(leaves)):
            leaf = leaves.popleft()
            for neighbor in graph[leaf]:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    leaves.append(neighbor)
            total_nodes -= 1
        operations += 1

    print(operations)

solve()