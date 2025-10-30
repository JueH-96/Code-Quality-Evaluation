import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
m = int(data[1])
edges = []
for i in range(m):
    u, v = int(data[2 + 2 * i]), int(data[3 + 2 * i])
    edges.append((u, v))

# Check if the graph is a tree
def is_tree(n, edges):
    if len(edges) != n - 1:
        return False
    parent = list(range(n + 1))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for u, v in edges:
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return False
        parent[root_u] = root_v
    return True

if not is_tree(n, edges):
    print("No")
else:
    print("Yes")
    # Assign values to vertices
    values = [1] * (n + 1)
    for u, v in edges:
        values[v] = values[u] ^ 1
    print(" ".join(map(str, values[1:])))