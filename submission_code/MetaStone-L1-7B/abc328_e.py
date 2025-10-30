import itertools

def find(u, parent):
    while parent[u] != u:
        parent[u] = parent[parent[u]]
        u = parent[u]
    return u

n, m, k = map(int, input().split())

edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

min_mod = k

for combo in itertools.combinations(edges, n-1):
    parent = list(range(n + 1))  # 0 to n
    for u, v, w in combo:
        ru = find(u, parent)
        rv = find(v, parent)
        if ru != rv:
            parent[rv] = ru
    # Check if all nodes are connected
    root = find(1, parent)
    connected = True
    for node in range(2, n + 1):
        if find(node, parent) != root:
            connected = False
            break
    if connected:
        total = sum(w for u, v, w in combo)
        mod_val = total % k
        if mod_val < min_mod:
            min_mod = mod_val
            if mod_val == 0:
                print(0)
                exit()

print(min_mod)