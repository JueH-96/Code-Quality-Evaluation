import sys

def find(parent, u):
    if parent[u] != u:
        parent[u] = find(parent, parent[u])
    return parent[u]

def union(parent, rank, u, v):
    u_root = find(parent, u)
    v_root = find(parent, v)
    if u_root == v_root:
        return
    if rank[u_root] < rank[v_root]:
        parent[u_root] = v_root
    else:
        parent[v_root] = u_root
    if rank[u_root] == rank[v_root]:
        rank[u_root] += 1

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    operations = []
    for _ in range(M):
        K_i = int(data[ptr])
        ptr += 1
        C_i = int(data[ptr])
        ptr += 1
        S_i = []
        for _ in range(K_i):
            S_i.append(int(data[ptr]))
            ptr += 1
        operations.append((C_i, S_i))
    # Sort operations by C_i
    operations.sort(key=lambda x: x[0])
    # Initialize union-find
    parent = [i for i in range(N+1)]
    rank = [0] * (N+1)
    total_weight = 0
    for op in operations:
        C_i, S_i = op
        # Find unique parents
        unique_parents = set()
        for u in S_i:
            p = find(parent, u)
            unique_parents.add(p)
        k = len(unique_parents)
        if k >= 2:
            total_weight += (k - 1) * C_i
            # Merge all components together
            root = unique_parents.pop()
            for p in unique_parents:
                union(parent, rank, p, root)
    # Check if graph is connected
    root = -1
    for u in range(1, N+1):
        p = find(parent, u)
        if root == -1:
            root = p
        elif p != root:
            print(-1)
            return
    print(total_weight)

if __name__ == '__main__':
    main()