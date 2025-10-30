def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    groups = [[] for _ in range(N)]
    for _ in range(M):
        A = int(data[idx]) - 1
        idx += 1
        B = int(data[idx]) - 1
        idx += 1
        C = int(data[idx])
        idx += 1
        groups[A].append((B, C))
    
    parent = list(range(N))
    rank = [1] * N
    xor_to_parent = [0] * N
    
    def find(u):
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        root = u
        reversed_path = path[::-1]
        accumulated_xor = 0
        for node in reversed_path:
            accumulated_xor ^= xor_to_parent[node]
            xor_to_parent[node] = accumulated_xor
            parent[node] = root
        return (root, accumulated_xor)
    
    def union(u, v, p):
        root_u, x_u = find(u)
        root_v, x_v = find(v)
        if root_u == root_v:
            return (x_u ^ x_v) == p
        if rank[root_u] > rank[root_v]:
            root_u, root_v = root_v, root_u
            x_u, x_v = x_v, x_u
            u, v = v, u
        parent[root_u] = root_v
        xor_to_parent[root_u] = x_u ^ p ^ x_v
        if rank[root_u] == rank[root_v]:
            rank[root_v] += 1
        return True
    
    conflict = False
    for A in range(N):
        lst = groups[A]
        m = len(lst)
        for i in range(m - 1):
            B1, C1 = lst[i]
            B2, C2 = lst[i + 1]
            p = C1 ^ C2
            if not union(B1, B2, p):
                conflict = True
    
    if conflict:
        print(-1)
    else:
        h = []
        for i in range(N):
            root, x = find(i)
            h.append(x)
        c = ['0'] * N
        for A in range(N):
            if groups[A]:
                B, C = groups[A][0]
                c_val = h[A] ^ h[B] ^ C
                c[A] = str(c_val)
        print(''.join(c))

if __name__ == "__main__":
    main()