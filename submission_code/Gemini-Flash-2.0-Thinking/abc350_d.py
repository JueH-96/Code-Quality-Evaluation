def solve():
    n, m = map(int, input().split())
    adj = [set() for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].add(v)
        adj[v].add(u)

    visited = [False] * n
    total_operations = 0

    for i in range(n):
        if not visited[i]:
            component_nodes = 0
            component_edges = 0
            stack = [i]
            visited[i] = True
            component_users = set()

            while stack:
                u = stack.pop()
                component_nodes += 1
                component_users.add(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)

            for u in component_users:
                for v in adj[u]:
                    if u < v and v in component_users:
                        component_edges += 1

            max_edges = component_nodes * (component_nodes - 1) // 2
            operations = max_edges - component_edges
            total_operations += operations

    print(total_operations)

solve()