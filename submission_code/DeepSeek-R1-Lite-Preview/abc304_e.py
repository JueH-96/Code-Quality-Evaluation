def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0

    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1

    parent = [i for i in range(N + 1)]
    rank = [1] * (N + 1)
    forbidden = [set() for _ in range(N + 1)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
            forbidden[root_y].update(forbidden[root_x])
            forbidden[root_y].discard(root_y)
        else:
            parent[root_y] = root_x
            forbidden[root_x].update(forbidden[root_y])
            forbidden[root_x].discard(root_x)
            if rank[root_x] == rank[root_y]:
                rank[root_x] += 1

    for _ in range(M):
        u = int(input[idx]); idx += 1
        v = int(input[idx]); idx += 1
        union(u, v)

    K = int(input[idx]); idx += 1
    for _ in range(K):
        x = int(input[idx]); idx += 1
        y = int(input[idx]); idx += 1
        root_x = find(x)
        root_y = find(y)
        forbidden[root_x].add(root_y)
        forbidden[root_y].add(root_x)

    Q = int(input[idx]); idx += 1
    output = []
    for _ in range(Q):
        p = int(input[idx]); idx += 1
        q = int(input[idx]); idx += 1
        root_p = find(p)
        root_q = find(q)
        if root_p == root_q:
            output.append("Yes")
        else:
            if root_q in forbidden[root_p] or root_p in forbidden[root_q]:
                output.append("No")
            else:
                output.append("Yes")
                union(p, q)
    print('
'.join(output))

if __name__ == '__main__':
    main()