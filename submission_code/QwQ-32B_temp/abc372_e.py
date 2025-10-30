import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())

    parent = list(range(N + 1))
    size = [1] * (N + 1)
    top = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        top[i] = [i]

    def find(u):
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        for v in path:
            parent[v] = u
        return u

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return
        if size[root_u] < size[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        size[root_u] += size[root_v]
        merged = top[root_u] + top[root_v]
        merged.sort(reverse=True)
        top[root_u] = merged[:10]

    for _ in range(Q):
        parts = sys.stdin.readline().split()
        query = list(map(int, parts))
        if query[0] == 1:
            u = query[1]
            v = query[2]
            union(u, v)
        else:
            v = query[1]
            k = query[2]
            root = find(v)
            lst = top[root]
            if len(lst) < k:
                print(-1)
            else:
                print(lst[k-1])

if __name__ == "__main__":
    main()