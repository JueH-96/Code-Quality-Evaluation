def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))

    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    count = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if j in adj[i]:
                continue

            # Create a new graph with the added edge
            new_adj = [list(neighbors) for neighbors in adj]
            new_adj[i].append(j)
            new_adj[j].append(i)

            # Find the cycle
            cycle = find_cycle(new_adj, i, j)

            if cycle is None:
                continue

            # Check if the graph is simple and all vertices in the cycle have degree 3
            is_valid = True
            for node in cycle:
                if len(new_adj[node]) != 3:
                    is_valid = False
                    break

            if is_valid:
                count += 1

    print(count)

def find_cycle(adj, start_node, end_node):
    n = len(adj) - 1
    visited = [False] * (n + 1)
    parent = [0] * (n + 1)
    cycle_start = -1
    cycle_end = -1

    def dfs(node, par):
        nonlocal cycle_start, cycle_end
        visited[node] = True
        parent[node] = par

        for neighbor in adj[node]:
            if neighbor == par:
                continue

            if visited[neighbor]:
                cycle_start = neighbor
                cycle_end = node
                return True
            
            if dfs(neighbor, node):
                return True
        
        return False

    if dfs(start_node, 0):
        cycle = []
        curr = cycle_end
        cycle.append(curr)
        while curr != cycle_start:
            curr = parent[curr]
            cycle.append(curr)
        
        return cycle
    else:
        return None

solve()