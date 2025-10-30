import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1
isolated_count = N
deg = [0] * (N + 1)
adj = [set() for _ in range(N + 1)]
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        if deg[u] == 0:
            isolated_count -= 1
        if deg[v] == 0:
            isolated_count -= 1
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    elif query_type == 2:
        v_rem = int(data[index])
        index += 1
        neighbors = list(adj[v_rem])
        if deg[v_rem] > 0:
            isolated_count += 1
        for neigh in neighbors:
            adj[neigh].remove(v_rem)
            adj[v_rem].remove(neigh)
            deg[neigh] -= 1
            if deg[neigh] == 0:
                isolated_count += 1
        deg[v_rem] = 0
    print(isolated_count)