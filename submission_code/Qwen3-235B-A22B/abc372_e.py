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
    top = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        top[i] = [i]

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    output = []
    for _ in range(Q):
        t = data[idx]
        idx += 1
        if t == '1':
            u = int(data[idx])
            idx += 1
            v = int(data[idx])
            idx += 1
            root_u = find(u)
            root_v = find(v)
            if root_u == root_v:
                continue
            if size[root_u] < size[root_v]:
                root_u, root_v = root_v, root_u
            parent[root_v] = root_u
            size[root_u] += size[root_v]
            merged = top[root_u] + top[root_v]
            merged.sort(reverse=True)
            top[root_u] = merged[:10]
        else:
            v = int(data[idx])
            idx += 1
            k = int(data[idx])
            idx += 1
            root_v = find(v)
            if k <= len(top[root_v]):
                output.append(str(top[root_v][k-1]))
            else:
                output.append('-1')
    
    print('
'.join(output))

if __name__ == "__main__":
    main()