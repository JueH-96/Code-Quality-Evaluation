import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    parent = list(range(N + 1))
    size = [1] * (N + 1)
    top = [[i] for i in range(N + 1)]

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for _ in range(Q):
        parts = sys.stdin.readline().split()
        if not parts:
            continue
        if parts[0] == '1':
            u = int(parts[1])
            v = int(parts[2])
            u_root = find(u)
            v_root = find(v)
            if u_root == v_root:
                continue
            if size[u_root] < size[v_root]:
                u_root, v_root = v_root, u_root
            parent[v_root] = u_root
            size[u_root] += size[v_root]
            merged = sorted(top[u_root] + top[v_root], reverse=True)[:10]
            top[u_root] = merged
        else:
            v = int(parts[1])
            k = int(parts[2])
            root = find(v)
            if size[root] < k:
                print(-1)
            else:
                print(top[root][k-1])

if __name__ == "__main__":
    main()