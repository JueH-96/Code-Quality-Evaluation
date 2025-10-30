import sys

def solve():
    n, q = map(int, sys.stdin.buffer.readline().split())
    adj = {i: set() for i in range(1, n + 1)}

    for _ in range(q):
        query = list(map(int, sys.stdin.buffer.readline().split()))
        query_type = query[0]

        if query_type == 1:
            u, v = query[1], query[2]
            adj[u].add(v)
            adj[v].add(u)
        elif query_type == 2:
            v, k = query[1], query[2]
            connected_vertices = sorted(list(adj[v] | {v}), reverse=True)
            if len(connected_vertices) < k:
                print(-1)
            else:
                print(connected_vertices[k - 1])

solve()