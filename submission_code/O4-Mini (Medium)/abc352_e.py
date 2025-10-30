import sys
import threading

def main():
    import sys

    data = sys.stdin
    line = data.readline().split()
    if not line:
        print(-1)
        return
    n, m = map(int, line)
    ops = []
    # Read operations
    for _ in range(m):
        parts = data.readline().split()
        while len(parts) < 2:
            parts += data.readline().split()
        k, c = map(int, parts)
        # read k vertices
        verts = []
        while len(verts) < k:
            verts.extend(map(int, data.readline().split()))
        # store (weight, verts_list)
        ops.append((c, verts))
    # sort by weight ascending for Kruskal
    ops.sort(key=lambda x: x[0])

    # DSU
    parent = list(range(n+1))
    size = [1] * (n+1)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        # union by size
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    mst_weight = 0
    edges_used = 0

    # Perform Kruskal: for each clique, connect it as a star centered at first vertex
    for cost, verts in ops:
        if edges_used == n-1:
            break
        if not verts:
            continue
        center = verts[0]
        for v in verts[1:]:
            if union(center, v):
                mst_weight += cost
                edges_used += 1
                if edges_used == n-1:
                    break

    # After processing all, check connectivity
    if edges_used < n-1:
        print(-1)
    else:
        print(mst_weight)

if __name__ == "__main__":
    main()