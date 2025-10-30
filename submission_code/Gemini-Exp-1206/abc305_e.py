import collections

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        edges.append(tuple(map(int, input().split())))
    guards = []
    for _ in range(k):
        guards.append(tuple(map(int, input().split())))

    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    guarded = [False] * (n + 1)
    
    for start_node, stamina in guards:
        q = collections.deque([(start_node, 0)])
        visited = [False] * (n + 1)
        visited[start_node] = True

        while q:
            curr_node, dist = q.popleft()
            if dist <= stamina:
                guarded[curr_node] = True
                for neighbor in adj[curr_node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append((neighbor, dist + 1))

    guarded_vertices = [i for i, val in enumerate(guarded) if val and i > 0]
    print(len(guarded_vertices))
    print(*guarded_vertices)

solve()