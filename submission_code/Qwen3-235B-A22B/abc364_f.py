import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    queries = []
    for i in range(Q):
        L, R, C = map(int, sys.stdin.readline().split())
        queries.append((L, R, C, i+1))  # i+1 is new node N + i+1?

    # Step 1: Check coverage of [1..N]
    intervals = sorted([(q[0], q[1]) for q in queries])
    merged = []
    for L, R in intervals:
        if not merged:
            merged.append([L, R])
        else:
            last_L, last_R = merged[-1]
            if L <= last_R + 1:
                merged[-1][1] = max(last_R, R)
            else:
                merged.append([L, R])
    ok = True
    if not merged:
        ok = False
    else:
        if merged[0][0] > 1:
            ok = False
        elif merged[-1][1] < N:
            ok = False
        else:
            for i in range(1, len(merged)):
                if merged[i][0] > merged[i-1][1] + 1:
                    ok = False
                    break
    if not ok:
        print(-1)
        return

    # Sort queries by cost
    queries.sort(key=lambda x: x[2])

    # DSU for all nodes (original and new)
    class DSU:
        def __init__(self, size):
            self.parent = list(range(size + 1))  # 0-based or 1-based?
            self.rank = [1] * (size + 1)

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root == y_root:
                return False
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1
            return True

    max_node = N + Q + 1
    dsu = DSU(max_node)

    # DSU for original nodes to track next available
    par_original = list(range(N + 2))  # 1-based to N

    def find(u):
        if u > N:
            return u
        while True:
            p = par_original[u]
            if p == u:
                return u
            u = p
            if u > N:
                return u

    total_cost = 0

    for L, R, C, idx in queries:
        v= N + idx
        count = 0
        u = find(L)
        while u <= R and u <= N:
            if dsu.find(u) != dsu.find(v):
                dsu.union(u, v)
                total_cost += C
                count += 1
            # Mark u as processed by moving its pointer
            par_original[u] = u + 1
            u = find(u + 1)
        if count == 0:
            print(-1)
            return

    # Check all original nodes are connected
    root = dsu.find(1)
    for u in range(2, N+1):
        if dsu.find(u) != root:
            print(-1)
            return
    # Check all new nodes are connected
    for i in range(1, Q+1):
        v = N + i
        if dsu.find(v) != root:
            print(-1)
            return

    print(total_cost)

if __name__ == "__main__":
    main()