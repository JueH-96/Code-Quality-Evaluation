from collections import defaultdict

def solve():
    N, Q = map(int, input().split())
    graph = defaultdict(set)
    isolated_vertices = set(range(1, N+1))

    for _ in range(Q):
        query = input().split()
        if query[0] == '1':
            u, v = map(int, query[1:])
            if u in isolated_vertices:
                isolated_vertices.remove(u)
            if v in isolated_vertices:
                isolated_vertices.remove(v)
            graph[u].add(v)
            graph[v].add(u)
        else:
            v = int(query[1])
            if v in isolated_vertices:
                print(len(isolated_vertices))
                continue
            for neighbor in graph[v]:
                graph[neighbor].remove(v)
                if len(graph[neighbor]) == 0:
                    isolated_vertices.add(neighbor)
            graph[v].clear()
            isolated_vertices.add(v)
            print(len(isolated_vertices))

solve()