from collections import defaultdict

N, Q = map(int, input().split())
edges = defaultdict(set)
isolated = set(range(1, N+1))
result = []

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        u, v = query[1], query[2]
        edges[u].add(v)
        edges[v].add(u)
        if u in isolated:
            isolated.remove(u)
        if v in isolated:
            isolated.remove(v)
    elif query[0] == 2:
        v = query[1]
        isolated.add(v)
        for u in edges[v]:
            edges[u].remove(v)
            if not edges[u]:
                isolated.add(u)
        edges[v].clear()
    result.append(len(isolated))

print('
'.join(map(str, result)))