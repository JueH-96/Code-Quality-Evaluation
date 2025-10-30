import collections

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    guards = []
    for _ in range(k):
        p, h = map(int, input().split())
        guards.append((p, h))

    graph = collections.defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    guarded = set()
    for guard_pos, guard_stamina in guards:
        q = collections.deque([(guard_pos, 0)])
        visited = {guard_pos}
        while q:
            curr_node, dist = q.popleft()
            if dist <= guard_stamina:
                guarded.add(curr_node)
                for neighbor in graph[curr_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

    guarded_list = sorted(list(guarded))
    print(len(guarded_list))
    print(*guarded_list)

solve()