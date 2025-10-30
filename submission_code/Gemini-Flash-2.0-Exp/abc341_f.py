def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u - 1, v - 1))
    weights = list(map(int, input().split()))
    amounts = list(map(int, input().split()))

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    total_operations = 0
    pieces = sum(amounts)

    while pieces > 0:
        best_vertex = -1
        for i in range(n):
            if amounts[i] > 0:
                best_vertex = i
                break

        if best_vertex == -1:
            break

        amounts[best_vertex] -= 1
        pieces -= 1
        total_operations += 1

        neighbors = adj[best_vertex]
        eligible_neighbors = []
        for neighbor in neighbors:
            if weights[neighbor] < weights[best_vertex]:
                eligible_neighbors.append(neighbor)

        for neighbor in eligible_neighbors:
            amounts[neighbor] += 1
            pieces += 1

    print(total_operations)

solve()