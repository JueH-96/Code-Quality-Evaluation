n = int(input())
edges = []
graph = [[] for _ in range(n)]
for _ in range(n - 1):
    u, v = map(int, input().split())
    u -= 1
    v -= 1
    edges.append((u, v))
    graph[u].append(v)
    graph[v].append(u)

count = 0
for i in range(n):
    for j in range(i + 1, n):
        is_simple = True
        cycle_nodes = set()
        visited = [False] * n
        q = [(i, -1)]
        visited[i] = True
        while q:
            curr, parent = q.pop(0)
            if curr == j:
                cycle_nodes.add(i)
                cycle_nodes.add(j)
                curr_node = parent
                while curr_node != i:
                    cycle_nodes.add(curr_node)
                    for neighbor in graph[curr_node]:
                        if neighbor != parent and not visited[neighbor]:
                            visited[neighbor] = True
                            q.append((neighbor, curr_node))
                            break
                    curr_node = parent
                    parent = -1
                    for neighbor in graph[curr_node]:
                        if neighbor == curr:
                            parent = neighbor
                            break
                break
            for neighbor in graph[curr]:
                if neighbor != parent and not visited[neighbor]:
                    visited[neighbor] = True
                    q.append((neighbor, curr))

        degree_3 = True
        for node in cycle_nodes:
            if len(graph[node]) + (1 if node == i or node == j else 0) != 3:
                degree_3 = False
                break
        if degree_3:
            count += 1

print(count)