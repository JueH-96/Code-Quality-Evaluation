def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    if a == b:
        print("Yes")
        return

    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, min(n, i + k + 1)):
            union(i, j)

    components = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    for root in components:
        indices = components[root]
        a_vals = sorted([a[i] for i in indices])
        b_vals = sorted([b[i] for i in indices])
        if a_vals != b_vals:
            print("No")
            return

    print("Yes")

t = int(input())
for _ in range(t):
    solve()