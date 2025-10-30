class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))

    k = int(input())
    bad_pairs = []
    for _ in range(k):
        x, y = map(int, input().split())
        bad_pairs.append((x, y))

    q = int(input())
    queries = []
    for _ in range(q):
        p, q_val = map(int, input().split())
        queries.append((p, q_val))

    for p_query, q_query in queries:
        dsu = DSU(n)
        for u, v in edges:
            dsu.union(u, v)

        is_good = True
        for x, y in bad_pairs:
            root_x = dsu.find(x)
            root_y = dsu.find(y)
            root_p = dsu.find(p_query)
            root_q = dsu.find(q_query)

            if (root_x == root_p and root_y == root_q) or (root_x == root_q and root_y == root_p):
                is_good = False
                break
            elif dsu.find(x) == dsu.find(y):
                continue # Should not happen based on problem statement

        if is_good:
            print("Yes")
        else:
            print("No")

solve()