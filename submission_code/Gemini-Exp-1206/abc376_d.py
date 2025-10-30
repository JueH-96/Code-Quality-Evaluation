import collections

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        edges.append(tuple(map(int, input().split())))

    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    q = collections.deque([(1, [1])])
    min_cycle_len = float('inf')

    while q:
        curr, path = q.popleft()

        if curr == 1 and len(path) > 1:
            min_cycle_len = min(min_cycle_len, len(path) - 1)
            continue

        for neighbor in adj[curr]:
            if neighbor not in path or (neighbor == 1 and len(path) > 1):
                q.append((neighbor, path + [neighbor]))

    if min_cycle_len == float('inf'):
        print(-1)
    else:
        print(min_cycle_len)

solve()