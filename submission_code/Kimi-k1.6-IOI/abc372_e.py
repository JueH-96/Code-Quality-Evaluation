def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    parent = list(range(N+1))  # 1-based indexing
    size = [1] * (N + 1)
    top_list = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        top_list[i] = [i]
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    output = []
    for _ in range(Q):
        query = data[idx]
        idx += 1
        if query == '1':
            u = int(data[idx])
            idx += 1
            v = int(data[idx])
            idx += 1
            # Union u and v
            ru = find(u)
            rv = find(v)
            if ru != rv:
                # Merge smaller into larger
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                size[ru] += size[rv]
                # Merge top lists
                merged = top_list[ru] + top_list[rv]
                merged.sort(reverse=True)
                top_list[ru] = merged[:10]
        else:
            v = int(data[idx])
            idx += 1
            k = int(data[idx])
            idx += 1
            # Query
            rv = find(v)
            if size[rv] < k:
                output.append(-1)
            else:
                output.append(top_list[rv][k-1])
    
    print('
'.join(map(str, output)))

if __name__ == "__main__":
    main()