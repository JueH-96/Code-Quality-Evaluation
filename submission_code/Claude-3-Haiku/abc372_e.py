from collections import defaultdict

N, Q = map(int, input().split())
graph = defaultdict(list)

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        u, v = query[1], query[2]
        graph[u].append(v)
        graph[v].append(u)
    else:
        v, k = query[1], query[2]
        neighbors = graph[v]
        neighbors.sort(reverse=True)
        if len(neighbors) < k:
            print(-1)
        else:
            print(neighbors[k-1])