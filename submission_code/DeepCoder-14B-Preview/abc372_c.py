def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    parent = list(range(N + 1))
    size = [1] * (N + 1)
    top = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        top[i] = [i]
    
    def find(u):
        root = u
        while parent[root] != root:
            root = parent[root]
        while u != root:
            next_u = parent[u]
            parent[u] = root
            u = next_u
        return root
    
    output = []
    for _ in range(Q):
        query_type = int(data[idx])
        idx += 1
        if query_type == 1:
            u = int(data[idx])
            idx += 1
            v = int(data[idx])
            idx += 1
            ru = find(u)
            rv = find(v)
            if ru == rv:
                continue
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            size[ru] += size[rv]
            merged = top[ru] + top[rv]
            merged.sort(reverse=True)
            merged = merged[:10]
            top[ru] = merged
        else:
            v = int(data[idx])
            idx += 1
            k = int(data[idx])
            idx += 1
            root = find(v)
            if len(top[root]) >= k:
                output.append(str(top[root][k-1]))
            else:
                output.append("-1")
    
    print('
'.join(output))

if __name__ == '__main__':
    main()