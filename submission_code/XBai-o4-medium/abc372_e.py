import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    parent = list(range(N + 1))
    size = [1] * (N + 1)
    top = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        top[i] = [i]

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for _ in range(Q):
        query_type = int(input[ptr])
        ptr += 1
        if query_type == 1:
            u = int(input[ptr])
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            u_root = find(u)
            v_root = find(v)
            if u_root == v_root:
                continue
            if size[u_root] < size[v_root]:
                u_root, v_root = v_root, u_root
            parent[v_root] = u_root
            size[u_root] += size[v_root]
            merged = top[u_root] + top[v_root]
            merged.sort(reverse=True)
            top[u_root] = merged[:10]
        else:
            v = int(input[ptr])
            ptr += 1
            k = int(input[ptr])
            ptr += 1
            root_v = find(v)
            if size[root_v] < k:
                print(-1)
            else:
                print(top[root_v][k-1])

if __name__ == "__main__":
    main()