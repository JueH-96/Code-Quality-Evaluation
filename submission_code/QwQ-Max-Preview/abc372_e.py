import sys

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    parent = list(range(N + 1))
    size = [1] * (N + 1)
    top10 = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        top10[i] = [i]
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
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
            a = find(u)
            b = find(v)
            if a != b:
                if size[a] < size[b]:
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]
                merged = top10[a] + top10[b]
                merged.sort(reverse=True)
                top10[a] = merged[:10]
        else:
            v = int(data[idx])
            idx += 1
            k = int(data[idx])
            idx += 1
            root = find(v)
            if size[root] < k:
                output.append('-1')
            else:
                output.append(str(top10[root][k-1]))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()