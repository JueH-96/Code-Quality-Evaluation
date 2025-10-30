def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    parent = [i for i in range(N + 1)]
    rank = [1] * (N + 1)
    size = [1] * (N + 1)
    top_k = [[i] for i in range(N + 1)]

    output = []

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
                size[root_v] += size[root_u]
                merged = top_k[root_v] + top_k[root_u]
                merged.sort(reverse=True)
                top_k[root_v] = merged[:10]
            else:
                parent[root_v] = root_u
                size[root_u] += size[root_v]
                merged = top_k[root_u] + top_k[root_v]
                merged.sort(reverse=True)
                top_k[root_u] = merged[:10]
                if rank[root_u] == rank[root_v]:
                    rank[root_u] += 1

    for _ in range(Q):
        query_type = int(input[ptr])
        ptr += 1
        if query_type == 1:
            u = int(input[ptr])
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            union(u, v)
        else:
            v = int(input[ptr])
            ptr += 1
            k = int(input[ptr])
            ptr += 1
            root = find(v)
            if size[root] >= k:
                output.append(top_k[root][k - 1])
            else:
                output.append(-1)

    print('
'.join(map(str, output)))

if __name__ == "__main__":
    main()