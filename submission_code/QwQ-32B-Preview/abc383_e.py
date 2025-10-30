import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    from itertools import repeat, chain
    import heapq
    import sys
    sys.setrecursionlimit(1 << 25)
    N_M_K = sys.stdin.readline().split()
    N = int(N_M_K[0])
    M = int(N_M_K[1])
    K = int(N_M_K[2])
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        u -= 1
        v -= 1
        edges.append((w, u, v))
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    A = [a - 1 for a in A]
    B = [b - 1 for b in B]

    # Union-Find
    parent = list(range(N))
    rank = [0] * N
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:
            return
        if rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[x] = y
            if rank[x] == rank[y]:
                rank[y] += 1

    # Kruskal's algorithm to build MST
    MST = []
    edges.sort()
    for w, u, v in edges:
        if find(u) != find(v):
            MST.append((w, u, v))
            union(u, v)
    
    # Initialize counts
    A_count = [0] * N
    B_count = [0] * N
    B_available = [0] * N
    for a in A:
        A_count[a] = 1
    for b in B:
        B_count[b] = 1
        B_available[b] = 1
    
    # DSU functions
    def get_root(x):
        if parent[x] != x:
            parent[x] = get_root(parent[x])
        return parent[x]
    
    total_sum = 0
    assigned_B = 0
    for w, u, v in MST:
        root_u = get_root(u)
        root_v = get_root(v)
        if root_u == root_v:
            continue
        A_u = A_count[root_u]
        B_u = B_available[root_u]
        A_v = A_count[root_v]
        B_v = B_available[root_v]
        
        # Assign min(A_u, B_v) pairs with weight w
        assignments1 = min(A_u, B_v)
        total_sum += assignments1 * w
        B_available[root_v] -= assignments1
        # Assign min(A_v, B_u) pairs with weight w
        assignments2 = min(A_v, B_u)
        total_sum += assignments2 * w
        B_available[root_u] -= assignments2
        
        # Merge components
        if rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
            A_count[root_u] += A_count[root_v]
            B_available[root_u] += B_available[root_v]
        else:
            parent[root_u] = root_v
            A_count[root_v] += A_count[root_u]
            B_available[root_v] += B_available[root_u]
            if rank[root_u] == rank[root_v]:
                rank[root_v] += 1
        assigned_B += assignments1 + assignments2
        if assigned_B == K:
            break
    
    print(total_sum)

if __name__ == "__main__":
    main()