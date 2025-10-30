import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1

    top = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        top[i] = [i]
    parent = list(range(N + 1))
    size = [1] * (N + 1)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    output = []
    for _ in range(Q):
        query_type = data[idx]
        idx += 1
        if query_type == '1':
            u = int(data[idx])
            idx += 1
            v = int(data[idx])
            idx += 1
            ru = find(u)
            rv = find(v)
            if ru != rv:
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                # Merge rv into ru
                combined = top[ru] + top[rv]
                combined.sort(reverse=True)
                top[ru] = combined[:10]
                size[ru] += size[rv]
                parent[rv] = ru
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

if __name__ == "__main__":
    main()