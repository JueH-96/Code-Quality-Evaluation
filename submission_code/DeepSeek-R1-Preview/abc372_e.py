import sys
input = sys.stdin.read
data = input().split()

idx = 0
N = int(data[idx])
idx += 1
Q = int(data[idx])
idx += 1

parent = list(range(N + 1))
size = [1] * (N + 1)
top = [[] for _ in range(N + 1)]
for i in range(1, N + 1):
    top[i] = [i]

def find(u):
    path = []
    while parent[u] != u:
        path.append(u)
        u = parent[u]
    for v in path:
        parent[v] = u
    return u

for _ in range(Q):
    query_type = int(data[idx])
    idx += 1
    if query_type == 1:
        u = int(data[idx])
        idx += 1
        v = int(data[idx])
        idx += 1
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if size[root_u] < size[root_v]:
                root_u, root_v = root_v, root_u
            parent[root_v] = root_u
            size[root_u] += size[root_v]
            merged = top[root_u] + top[root_v]
            merged.sort(reverse=True)
            top[root_u] = merged[:10]
    else:
        v = int(data[idx])
        idx += 1
        k = int(data[idx])
        idx += 1
        root = find(v)
        if len(top[root]) >= k:
            print(top[root][k-1])
        else:
            print(-1)