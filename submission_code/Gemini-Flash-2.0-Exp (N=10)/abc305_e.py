from collections import deque

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        edges.append(list(map(int, input().split())))
    guards = []
    for _ in range(k):
        guards.append(list(map(int, input().split())))

    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    guarded_vertices = set()

    for guard_pos, guard_stamina in guards:
        q = deque([(guard_pos, 0)])
        visited = {guard_pos}
        
        while q:
            curr_node, dist = q.popleft()
            if dist <= guard_stamina:
                guarded_vertices.add(curr_node)
                for neighbor in adj[curr_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))
    
    sorted_guarded = sorted(list(guarded_vertices))
    print(len(sorted_guarded))
    print(*sorted_guarded)

solve()