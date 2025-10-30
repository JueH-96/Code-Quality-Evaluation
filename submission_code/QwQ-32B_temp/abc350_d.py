import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    edges = []
    parent = list(range(N + 1))
    size = [1] * (N + 1)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        if size[u_root] < size[v_root]:
            u_root, v_root = v_root, u_root
        parent[v_root] = u_root
        size[u_root] += size[v_root]

    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        edges.append((a, b))
        union(a, b)

    edge_count = [0] * (N + 1)
    for a, b in edges:
        root = find(a)
        edge_count[root] += 1

    result = 0
    for i in range(1, N + 1):
        root = find(i)
        if root == i:
            s = size[root]
            m_i = edge_count[root]
            result += s * (s - 1) // 2 - m_i

    print(result)

if __name__ == "__main__":
    main()