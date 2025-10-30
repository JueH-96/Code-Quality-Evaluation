import sys
import threading
def main():
    import sys
    data = sys.stdin
    line = data.readline()
    if not line:
        return
    N, M = map(int, line.split())
    ops = []
    # Read operations
    for _ in range(M):
        parts = data.readline().split()
        while not parts:
            parts = data.readline().split()
        K_i = int(parts[0])
        C_i = int(parts[1])
        arr = list(map(int, data.readline().split()))
        # convert to 0-based
        for j in range(K_i):
            arr[j] -= 1
        ops.append((C_i, arr))
    # Sort by edge weight
    ops.sort(key=lambda x: x[0])
    # DSU implementation
    parent = list(range(N))
    rank = [0] * N
    def find(x):
        # path-halving
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        # find roots
        x_root = x
        while parent[x_root] != x_root:
            parent[x_root] = parent[parent[x_root]]
            x_root = parent[x_root]
        y_root = y
        while parent[y_root] != y_root:
            parent[y_root] = parent[parent[y_root]]
            y_root = parent[y_root]
        if x_root == y_root:
            return False
        # union by rank
        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        else:
            parent[y_root] = x_root
            if rank[x_root] == rank[y_root]:
                rank[x_root] += 1
        return True

    total_cost = 0
    components = N

    # Kruskal-like: for each clique operation in ascending weight
    for cost, vertices in ops:
        if components == 1:
            break
        # collect the distinct DSU roots in this clique
        roots = set()
        for v in vertices:
            roots.add(find(v))
        if len(roots) <= 1:
            continue
        # connect all these roots with (len(roots)-1) edges of this cost
        roots_list = list(roots)
        base = roots_list[0]
        for r in roots_list[1:]:
            if union(base, r):
                total_cost += cost
                components -= 1
                if components == 1:
                    break

    # If not fully connected, answer is -1
    if components > 1:
        print(-1)
    else:
        print(total_cost)

# call main
if __name__ == "__main__":
    main()