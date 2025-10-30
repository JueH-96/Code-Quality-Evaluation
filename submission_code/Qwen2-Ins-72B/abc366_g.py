from collections import defaultdict

def solve(N, M, edges):
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    
    # Check for odd degree cycle
    visited = set()
    for v in range(1, N+1):
        if v in visited:
            continue
        stack = [v]
        path = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            path.append(node)
            for neighbor in graph[node]:
                stack.append(neighbor)
        # Check for odd degree cycle in the path
        degrees = {v: 0 for v in path}
        for v in path:
            degrees[v] = len(graph[v])
        for v in path:
            if degrees[v] % 2 == 1:
                cycle = [v]
                next_v = v
                while True:
                    for neighbor in graph[next_v]:
                        if neighbor in cycle:
                            continue
                        if degrees[neighbor] % 2 == 1:
                            cycle.append(neighbor)
                            next_v = neighbor
                            break
                    else:
                        break
                if len(cycle) % 2 == 1:
                    return "No"
    
    # Assign values
    values = [0] * (N+1)
    for v in range(1, N+1):
        if values[v] == 0:
            values[v] = 1
            stack = [v]
            while stack:
                node = stack.pop()
                for neighbor in graph[node]:
                    if values[neighbor] == 0:
                        values[neighbor] = values[node]
                        stack.append(neighbor)
                    else:
                        values[node] ^= values[neighbor]
    
    return "Yes
" + " ".join(map(str, values[1:]))


N, M = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(M)]
print(solve(N, M, edges))