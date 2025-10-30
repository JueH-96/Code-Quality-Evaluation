def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0

    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1

    parent = [i for i in range(N)]
    rank = [1] * N

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        else:
            parent[u_root] = v_root
            if rank[u_root] == rank[v_root]:
                rank[v_root] += 1

    for _ in range(M):
        u = int(data[idx]) - 1
        idx += 1
        v = int(data[idx]) - 1
        idx += 1
        union(u, v)

    K = int(data[idx])
    idx += 1

    forbidden = [set() for _ in range(N)]

    for _ in range(K):
        x = int(data[idx]) - 1
        idx += 1
        y = int(data[idx]) - 1
        idx += 1
        x_root = find(x)
        y_root = find(y)
        forbidden[x_root].add(y_root)
        forbidden[y_root].add(x_root)

    Q = int(data[idx])
    idx += 1

    results = []
    for _ in range(Q):
        p = int(data[idx]) - 1
        idx += 1
        q = int(data[idx]) - 1
        idx += 1
        p_root = find(p)
        q_root = find(q)
        if p_root == q_root:
            results.append("Yes")
        else:
            if q_root in forbidden[p_root] or p_root in forbidden[q_root]:
                results.append("No")
            else:
                results.append("Yes")
    
    print("
".join(results))

if __name__ == "__main__":
    main()