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

    parent = list(range(N + 2))  # 1-based to N
    color = list(range(N + 2))    # same as parent initially
    size = [1] * (N + 2)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for _ in range(Q):
        query = data[idx]
        idx += 1
        if query == '1':
            x = int(data[idx])
            idx += 1
            c = int(data[idx])
            idx += 1
            r = find(x)
            color[r] = c
            if c not in parent:
                parent[c] = r
                size[c] = 1
                root_c = find(c)
                if root_c != r:
                    if size[r] < size[root_c]:
                        parent[r] = root_c
                        size[root_c] += size[r]
                    else:
                        parent[root_c] = r
                        size[r] += size[root_c]
            else:
                root_c = find(c)
                if root_c != r:
                    if size[r] < size[root_c]:
                        parent[r] = root_c
                        size[root_c] += size[r]
                    else:
                        parent[root_c] = r
                        size[r] += size[root_c]
        else:
            c = int(data[idx])
            idx += 1
            print(size[c])

if __name__ == '__main__':
    main()