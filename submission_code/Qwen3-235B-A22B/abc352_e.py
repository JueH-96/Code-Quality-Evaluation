import sys

def main():
    sys.setrecursionlimit(1 << 25)
    data = list(map(int, sys.stdin.read().split()))
    ptr = 0
    N = data[ptr]
    ptr += 1
    M = data[ptr]
    ptr += 1

    ops = []
    for _ in range(M):
        K_i = data[ptr]
        ptr += 1
        C_i = data[ptr]
        ptr += 1
        A_list = data[ptr:ptr+K_i]
        ptr += K_i
        ops.append((C_i, A_list))
    
    # Sort operations by cost ascending
    ops.sort()

    # Initialize DSU
    parent = list(range(N + 2))  # 1-based indexing
    rank = [0] * (N + 2)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        # Union by rank
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        else:
            parent[v_root] = u_root
            if rank[u_root] == rank[v_root]:
                rank[u_root] += 1
        return True

    mst_cost = 0
    components = N

    for C_i, A_list in ops:
        roots = set()
        for node in A_list:
            roots.add(find(node))
        k = len(roots)
        if k > 1:
            mst_cost += (k - 1) * C_i
            components -= (k - 1)
            # Merge all into the first root
            roots = list(roots)
            target = roots[0]
            for r in roots[1:]:
                union(target, r)
    
    if components == 1:
        print(mst_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()